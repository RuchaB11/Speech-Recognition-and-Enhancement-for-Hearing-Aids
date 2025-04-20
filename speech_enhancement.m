function main()
    % Clear workspace and command window
    clc; close all; clear;

    % Check for required toolboxes first
    checkRequiredToolboxes();

    % Configuration - easily adjustable parameters
    config = struct();
    config.inputFolder =  'C:\Users\sathw\Documents\MATLAB\input_audio';  % Replace with your input directory
    config.outputFolder = 'C:\Users\sathw\Documents\MATLAB\enhanced_audio'; % Replace with your output directory
    config.visualizationFolder = 'C:\Users\sathw\Documents\MATLAB\audio_visualizations'; % Replace with your visualization directory
    config.target_fs = 16000;
    config.frame_length_ms = 40;  % Reduced for better time resolution
    config.frame_overlap_percent = 50; % Increased overlap for smoother reconstruction
    config.preemphasis_coeff = 0.97;

    % Advanced parameters - Tuned for hearing aid scenarios
    config.kalman = struct('Q', 1e-5, 'R', 0.05, 'initial_state', 0, 'initial_covar', 0.1); % Increased R for more stable tracking
    config.spectral = struct('alpha', 2.5, 'beta', 0.005); % Adjusted spectral subtraction parameters
    config.wavelet = struct('level', 5, 'family', 'db4', 'threshold_type', 's'); % Reduced wavelet level
    config.normalization = struct('target_peak', 0.95);

    % Flags for enabling/disabling techniques
    config.useWiener = true;
    config.useKalman = true;
    config.useSpectral = false; % Disabled due to ISTFT issues and potential artifacts. Can re-enable if fixed.
    config.useMedian = true;
    config.useWavelet = true;

    % Create output directories
    ensureDirectoryExists(config.outputFolder);
    ensureDirectoryExists(config.visualizationFolder);

    % Process all WAV files
    processAudioFiles(config);

    fprintf('Processing complete. Enhanced audio files saved to %s\n', config.outputFolder);
end

% =========================================================================
% Helper Functions
% =========================================================================

function checkRequiredToolboxes()
    required = {'Signal_Processing_Toolbox', 'Audio_Toolbox', 'Wavelet_Toolbox'}; % Added Wavelet
    missing = false;

    fprintf('Checking required toolboxes:\n');
    for i = 1:length(required)
        hasToolbox = license('test', required{i}) && ~isempty(ver(required{i})); % Check license and installation
        fprintf('  %s: %s\n', required{i}, ternary(hasToolbox, 'Installed', 'Missing'));
        if ~hasToolbox
            missing = true;
        end
    end

    if missing
        warning('Some required toolboxes are missing or not installed. The script may not function correctly.');
    end
end

function result = ternary(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

function ensureDirectoryExists(dirPath)
    if ~exist(dirPath, 'dir')
        try
            mkdir(dirPath);
            fprintf('Created directory: %s\n', dirPath);
        catch ME
            error('Could not create directory: %s\nError: %s', dirPath, ME.message);
        end
    end
end

% =========================================================================
% Core Processing Functions
% =========================================================================

function processAudioFiles(config)
    % Get list of WAV files
    fileList = dir(fullfile(config.inputFolder, '*.wav'));

    if isempty(fileList)
        error('No WAV files found in the input directory: %s', config.inputFolder);
    end

    % Process each file
    for i = 1:length(fileList)
        inputFile = fullfile(config.inputFolder, fileList(i).name);
        [~, name, ext] = fileparts(fileList(i).name);
        outputFile = fullfile(config.outputFolder, [name '_enhanced' ext]);
        visualizationPrefix = fullfile(config.visualizationFolder, name); % Define here for consistent naming

        fprintf('\nProcessing file %d of %d: %s\n', i, length(fileList), fileList(i).name);

        try
            processAudioFile(inputFile, outputFile, visualizationPrefix, config); % Pass prefix directly
        catch ME
            fprintf('Error processing file %s: %s\n', fileList(i).name, ME.message);
            % Display stack trace for debugging
            fprintf('Error details: %s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
            fprintf('Skipping to next file...\n');
        end
    end
end

function processAudioFile(inputFile, outputFile, visualizationPrefix, config)
    % Read and validate audio
    [originalAudio, originalFs] = validateAndReadAudio(inputFile);
    originalLength = length(originalAudio); % Store original length *before* any processing

    % Start processing pipeline
    tic;
    fs = originalFs;

    % Pre-processing stage
    [processedAudio, fs, preprocessInfo] = preprocessAudio(originalAudio, fs, config);

    % Speech enhancement stage
    % Pass original length for final length correction
    enhancedAudio = enhanceAudio(processedAudio, fs, config, originalLength);

    % Normalize output and write to file
    % Apply final normalization *after* ensuring correct length
    enhancedAudio = normalizeAudio(enhancedAudio, config.normalization.target_peak);
    audiowrite(outputFile, enhancedAudio, fs, 'BitsPerSample', 16);

    % Generate visualizations
    % Pass original length needed signals to visualization
    generateVisualizations(originalAudio, processedAudio, enhancedAudio, originalFs, fs, visualizationPrefix, preprocessInfo);

    processingTime = toc;
    fprintf('Processing completed in %.2f seconds (%.2fx real-time)\n', ...
        processingTime, processingTime / (length(originalAudio)/originalFs));
end

% =========================================================================
% Audio Input/Validation
% =========================================================================

function [audio, fs] = validateAndReadAudio(filename)
    try
        [audio, fs] = audioread(filename);

        % Check for empty audio
        if isempty(audio)
            error('Audio file is empty: %s', filename);
        end

        % Convert to mono if stereo
        if size(audio, 2) > 1
            audio = mean(audio, 2);
            fprintf('Converted stereo audio to mono.\n');
        end

        % Ensure double precision
        if ~isa(audio, 'double')
            audio = double(audio);
        end

        % Basic check for near-constant signal (silence) which might cause issues
        if std(audio) < 1e-9
            warning('Audio signal appears to be constant (silence or DC offset).');
        end

    catch ME
        error('Failed to read audio file: %s. Error: %s', filename, ME.message);
    end
end

% =========================================================================
% Pre-Processing Stage
% =========================================================================

function [processedAudio, fs, info] = preprocessAudio(audio, fs, config)
    % Store preprocessing information for visualization
    info = struct();
    info.originalAudio = audio; % Store for reference
    info.originalFs = fs;
    currentAudio = audio; % Work on a copy

    % Step 1: Resampling if needed
    if fs ~= config.target_fs
        fprintf('Resampling audio from %d Hz to %d Hz...\n', fs, config.target_fs);
        try
            currentAudio = resample(currentAudio, config.target_fs, fs);
            fs = config.target_fs;
            fprintf('Resampled audio to %d Hz.\n', fs);
        catch ME
            warning('Resampling failed: %s. Proceeding with original sample rate %d Hz.', ME.message, fs);
        end
    end
    info.resampledAudio = currentAudio; % Store potentially resampled audio

    % Step 2: Apply high-pass filter to remove low-frequency noise
    try
        [b, a] = designOptimalHighPassFilter(currentAudio, fs);
        filteredAudio = filter(b, a, currentAudio);
        info.filteredAudio = filteredAudio;
        info.filterCoeff = struct('b', b, 'a', a);
        currentAudio = filteredAudio; % Update working audio
        fprintf('Applied high-pass filter.\n');
    catch ME
        warning('High-pass filtering failed: %s. Skipping HPF.', ME.message);
        info.filteredAudio = currentAudio; % Store unfiltered if failed
        info.filterCoeff = [];
    end


    % Step 3: Pre-emphasis to boost high-frequency content
    emphasizedAudio = applyPreemphasis(currentAudio, config.preemphasis_coeff);
    currentAudio = emphasizedAudio; % Update working audio
    fprintf('Applied pre-emphasis (coeff=%.2f).\n', config.preemphasis_coeff);


    % Step 4: Normalization (Optional - usually done *after* enhancement)
    % It might be better to normalize *before* enhancement to ensure
    % consistent levels for algorithms, but also normalize *after*.
    % Let's normalize here for algorithm consistency.
    [normalizedAudio, normFactor] = normalizeAudio(currentAudio, config.normalization.target_peak);
    info.normalizationFactor = normFactor;
    currentAudio = normalizedAudio; % Update working audio
    fprintf('Applied pre-enhancement normalization (peak=%.2f).\n', config.normalization.target_peak);

    % Step 5: Frame the signal and apply windowing (FOR VISUALIZATION ONLY)
    % The actual enhancement algorithms will do their own framing if needed.
    [framedSegmentsForVis, windowFunc] = frameSignalForVis(currentAudio, fs, config);
    info.framedSegments = framedSegmentsForVis;
    info.window = windowFunc;

    processedAudio = currentAudio; % Return the fully pre-processed audio
end

function [b, a] = designOptimalHighPassFilter(audio, fs)
    % Analyze frequency content to determine optimal cutoff
    cutoffFreq = 80; % Default cutoff
    try
        nWin = min(length(audio), 1024);
        if nWin < 2 % Need at least 2 samples for pwelch
             warning('Audio too short for pwelch analysis in HPF design. Using default cutoff %d Hz.', cutoffFreq);
        else
            [pxx, f] = pwelch(audio, hamming(nWin), [], [], fs);

            % Find the noise floor (median of lower 10% of spectrum)
            noiseFloorIdx = max(1, round(length(pxx)*0.1));
            noiseFloor = median(pxx(1:noiseFloorIdx));

            % Find frequencies with significant energy above noise floor (e.g., 3x)
            significantIdx = find(pxx > noiseFloor * 3, 1, 'first'); % Find first significant index

            if ~isempty(significantIdx) && significantIdx > 2 && f(significantIdx) > 20
                % Use the frequency *below* the first significant peak, minimum 20Hz
                cutoffFreq = max(20, f(significantIdx-1));
                % Optional: Add upper bound, e.g., don't set cutoff > 150 Hz
                cutoffFreq = min(cutoffFreq, 150);
            else
                fprintf('Could not determine optimal cutoff from spectrum. Using default: %d Hz.\n', cutoffFreq);
            end
        end
    catch ME
        warning('pwelch analysis failed in HPF design: %s. Using default cutoff %d Hz.', ME.message, cutoffFreq);
    end

    % Design a 4th order Butterworth high-pass filter
    [b, a] = butter(4, cutoffFreq/(fs/2), 'high');
    fprintf('High-pass filter designed with cutoff frequency: %.1f Hz\n', cutoffFreq);
end

function audio_out = applyPreemphasis(audio_in, alpha)
    % Apply pre-emphasis filter: y[n] = x[n] - alpha*x[n-1]
    % Use filter function for clarity and potential Coder compatibility
    audio_out = filter([1 -alpha], 1, audio_in);
    % Handle potential length change/transient if needed, though filter usually preserves length
    % audio_out = audio_out(1:length(audio_in)); % Ensure length
end

function [normalizedAudio, factor] = normalizeAudio(audio, targetPeak)
    % Calculate normalization factor to reach target peak
    currentPeak = max(abs(audio(:))); % Use (:) for column vector safety

    if currentPeak > 1e-9 % Avoid division by zero or near-zero
        factor = targetPeak / currentPeak;
        normalizedAudio = audio * factor;
    else
        factor = 1; % No change if signal is silent
        normalizedAudio = audio;
    end
end

function [framedSegments, window] = frameSignalForVis(audio, fs, config)
    % Calculate frame parameters
    frameSizeSamples = round((config.frame_length_ms/1000) * fs);
    overlapSamples = round(frameSizeSamples * (config.frame_overlap_percent/100));
    hopSize = frameSizeSamples - overlapSamples;

    % Create Hamming window
    window = hamming(frameSizeSamples);

    % Extract first few frames for visualization only
    numFramesTotal = floor((length(audio) - overlapSamples) / hopSize); % More accurate frame count
    numFramesToVisualize = min(5, numFramesTotal);

    framedSegments = cell(1, numFramesToVisualize);
    if numFramesToVisualize < 1 % Handle case of very short audio
        return;
    end

    for i = 1:numFramesToVisualize
        startIdx = (i-1) * hopSize + 1;
        endIdx = startIdx + frameSizeSamples - 1;

        % Ensure end index is within bounds
        if endIdx > length(audio)
             frame = [audio(startIdx:end); zeros(endIdx - length(audio), 1)]; % Pad last frame
        else
            frame = audio(startIdx:endIdx);
        end

        % Check if frame is shorter than window (shouldn't happen with padding)
         if length(frame) ~= frameSizeSamples
             warning('Frame %d length (%d) does not match window size (%d) in visualization framing.', i, length(frame), frameSizeSamples);
             % Pad or truncate frame if necessary, though padding above should handle it
             frame = [frame(:); zeros(frameSizeSamples - length(frame), 1)]; % Ensure column and pad
             frame = frame(1:frameSizeSamples); % Truncate if too long
         end

        framedSegments{i} = frame .* window;
    end
end

% =========================================================================
% Enhancement Stage
% =========================================================================

function enhancedAudio = enhanceAudio(audio, fs, config, originalLength)
    % Apply all enhancement techniques
    try
        fprintf('Starting enhancement process...\n');
        len_in = length(audio); % Length before enhancement methods

        % Estimate noise characteristics from the signal
        noiseEstimate = estimateNoise(audio, fs);
        if isempty(noiseEstimate)
             warning('Noise estimation returned empty. Using a small segment of input as noise estimate.');
             noiseEstimate = audio(1:min(len_in, round(0.1*len_in))); % Fallback noise estimate
             if isempty(noiseEstimate) % If audio itself is empty
                 noiseEstimate = zeros(100,1); % Arbitrary small zero vector
             end
        end


        % Apply each enabled enhancement technique
        wienerEnhanced = audio; % Initialize with input audio
        kalmanEnhanced = audio;
        spectralEnhanced = audio;
        medianEnhanced = audio;
        waveletEnhanced = audio;
        active_methods = 0;

        if config.useWiener
            fprintf('Applying Wiener Filter...\n');
            try
                wienerEnhanced = applyWienerFilter(audio, noiseEstimate, fs, config);
                active_methods = active_methods + 1;
            catch ME
                fprintf('Wiener filter failed: %s\n', ME.message);
                % Keep wienerEnhanced as original 'audio'
            end
        end
        if config.useKalman
             fprintf('Applying Kalman Filter...\n');
            try
                kalmanEnhanced = applyKalmanFilter(audio, config.kalman);
                active_methods = active_methods + 1;
            catch ME
                fprintf('Kalman filter failed: %s\n', ME.message);
            end
        end
        if config.useSpectral
             fprintf('Applying Spectral Subtraction...\n');
            try
                spectralEnhanced = applySpectralSubtraction(audio, fs, noiseEstimate, config.spectral);
                 active_methods = active_methods + 1;
           catch ME
                fprintf('Spectral subtraction failed: %s\n', ME.message);
            end
        end
        if config.useMedian
             fprintf('Applying Median Filter...\n');
            try
                medianEnhanced = applyMedianFilter(audio);
                 active_methods = active_methods + 1;
           catch ME
                fprintf('Median filter failed: %s\n', ME.message);
            end
        end
        if config.useWavelet
             fprintf('Applying Wavelet Denoising...\n');
            try
                waveletEnhanced = applyWaveletDenoising(audio, config.wavelet);
                 active_methods = active_methods + 1;
           catch ME
                fprintf('Wavelet denoising failed: %s\n', ME.message);
                fprintf('Wavelet error details: %s\n', getReport(ME, 'extended', 'hyperlinks','off'));
            end
        end

        % Ensure all intermediate signals have the *same length* before combining
        % Use the length of the input 'audio' as the reference
        wienerEnhanced   = adjustSignalLength(wienerEnhanced, len_in);
        kalmanEnhanced   = adjustSignalLength(kalmanEnhanced, len_in);
        spectralEnhanced = adjustSignalLength(spectralEnhanced, len_in);
        medianEnhanced   = adjustSignalLength(medianEnhanced, len_in);
        waveletEnhanced  = adjustSignalLength(waveletEnhanced, len_in);

        % --- Adaptive Combination ---
        if active_methods > 0
             fprintf('Calculating adaptive weights...\n');
            % Ensure noise estimate is suitable for pwelch before calculating weights
            if length(noiseEstimate) >= 64 % Arbitrary minimum length for pwelch stability
                weights = calculateAdaptiveWeights(audio, noiseEstimate, fs, config);
            else
                warning('Noise estimate too short for reliable spectral analysis in weighting. Using default weights.');
                 % Fallback weights (distribute equally among enabled methods)
                 numEnabled = config.useWiener + config.useKalman + config.useSpectral + config.useMedian + config.useWavelet;
                 defaultWeight = ternary(numEnabled > 0, 1/numEnabled, 0);
                 weights = struct('wiener', config.useWiener*defaultWeight, ...
                                  'kalman', config.useKalman*defaultWeight, ...
                                  'spectral', config.useSpectral*defaultWeight, ...
                                  'median', config.useMedian*defaultWeight, ...
                                  'wavelet', config.useWavelet*defaultWeight);
            end

            fprintf('Combining enhanced signals...\n');
            % Combine enabled & weighted signals
            enhancedAudio = zeros(len_in, 1); % Initialize with zeros
            if config.useWiener;   enhancedAudio = enhancedAudio + weights.wiener * wienerEnhanced; end
            if config.useKalman;   enhancedAudio = enhancedAudio + weights.kalman * kalmanEnhanced; end
            if config.useSpectral; enhancedAudio = enhancedAudio + weights.spectral * spectralEnhanced; end
            if config.useMedian;   enhancedAudio = enhancedAudio + weights.median * medianEnhanced; end
            if config.useWavelet;  enhancedAudio = enhancedAudio + weights.wavelet * waveletEnhanced; end

        else
             warning('No enhancement methods were successfully applied or enabled. Returning pre-processed audio.');
             enhancedAudio = audio; % Return the input if no methods ran
        end

        % --- Post-Processing ---
        fprintf('Applying post-processing...\n');
        enhancedAudio = applyPostProcessing(enhancedAudio, fs);

        % --- Final Length Adjustment ---
        % Ensure final enhancedAudio has the *original input* length
        enhancedAudio = adjustSignalLength(enhancedAudio, originalLength);
        fprintf('Enhancement complete. Final length: %d samples.\n', length(enhancedAudio));

    catch ME
        fprintf('Error during enhancement processing stage: %s\n', ME.message);
        fprintf('Enhancement error details: %s\n', getReport(ME, 'extended', 'hyperlinks','off'));
        fprintf('Falling back to basic enhancement.\n');
        enhancedAudio = applyBasicEnhancement(audio, fs);
        % Ensure fallback also matches original length
        enhancedAudio = adjustSignalLength(enhancedAudio, originalLength);
    end
end

function noiseEstimate = estimateNoise(audio, fs)
    % Estimate noise using initial silence detection (improved)
    frameSizeMs = 20; % 20ms frames
    frameSize = round(frameSizeMs/1000 * fs);
    if frameSize < 1; frameSize = 1; end
    hopSize = floor(frameSize / 2); % 50% overlap

    numFrames = floor((length(audio) - frameSize) / hopSize) + 1;

    if numFrames < 5 % Need a few frames for reliable estimate
        warning('Audio too short for reliable noise estimation. Using first 10%% as estimate.');
        noiseEstLen = min(length(audio), round(0.1*length(audio)));
        noiseEstimate = audio(1:max(1, noiseEstLen)); % Ensure at least 1 sample
        return;
    end

    % Calculate energy (or RMS) of each frame
    frameEnergies = zeros(numFrames, 1);
    for i = 1:numFrames
        startIdx = (i-1)*hopSize + 1;
        endIdx = startIdx + frameSize - 1;
        % Ensure indices are valid
        endIdx = min(endIdx, length(audio));
        startIdx = min(startIdx, endIdx); % Prevent start > end for very short audio

        frame = audio(startIdx:endIdx);
        if ~isempty(frame)
             frameEnergies(i) = mean(frame.^2);
        else
             frameEnergies(i) = 0;
        end
    end

    % Find the frames with lowest energy (likely silence/noise)
    [sortedEnergies, sortedIndices] = sort(frameEnergies);

    % Use a percentile threshold to identify noise frames (e.g., lowest 15%)
    energyThreshold = sortedEnergies(max(1, floor(numFrames * 0.15))); % Threshold at 15th percentile

    % Collect frames below the threshold
    noiseFramesIndices = sortedIndices(frameEnergies(sortedIndices) <= energyThreshold);

    % Ensure we have at least a minimum number of frames (e.g., 5 or 10%)
    minNoiseFrames = max(3, floor(numFrames * 0.1));
    if length(noiseFramesIndices) < minNoiseFrames
       noiseFramesIndices = sortedIndices(1:minNoiseFrames); % Take the lowest energy frames if percentile method fails
    end

    % Concatenate noise frames into a single estimate
    noiseEstimate = [];
    for i = 1:length(noiseFramesIndices)
        frameIdx = noiseFramesIndices(i);
        startIdx = (frameIdx-1)*hopSize + 1;
        endIdx = min(startIdx + frameSize - 1, length(audio));
         if startIdx <= endIdx
             noiseEstimate = [noiseEstimate; audio(startIdx:endIdx)]; %#ok<AGROW>
         end
    end

    % Limit the length of the noise estimate (e.g., max 2 seconds)
    maxNoiseLen = min(length(audio), round(2 * fs));
    if length(noiseEstimate) > maxNoiseLen
        noiseEstimate = noiseEstimate(1:maxNoiseLen);
    elseif isempty(noiseEstimate) && length(audio)>0
         warning('Could not extract noise frames. Using first 10%% as estimate.');
         noiseEstLen = min(length(audio), round(0.1*length(audio)));
         noiseEstimate = audio(1:max(1, noiseEstLen));
     elseif isempty(noiseEstimate) && length(audio)==0
         noiseEstimate = zeros(100,1); % Fallback for empty input
    end

    fprintf('Estimated noise segment length: %d samples.\n', length(noiseEstimate));
end

function enhanced = applyWienerFilter(audio, noiseEstimate, fs, config)
    len_in = length(audio);
    frameSizeSamples = round((config.frame_length_ms/1000) * fs);
    overlapSamples = round(frameSizeSamples * (config.frame_overlap_percent/100));
    hopSize = frameSizeSamples - overlapSamples;
    fftSize = 2^nextpow2(frameSizeSamples); % Use power-of-2 FFT size
    numBinsOneSided = floor(fftSize/2) + 1; % Number of bins in one-sided spectrum

    if frameSizeSamples < 1
        error('Wiener Filter: Frame size must be at least 1 sample.');
    end

    window = hamming(frameSizeSamples, 'periodic');

    % Estimate noise power spectrum reliably (ensure it's one-sided)
    noisePowerSpectrum = estimateNoisePowerSpectrum(noiseEstimate, frameSizeSamples, window, fftSize);
    % --- Defensive Check ---
    if size(noisePowerSpectrum, 1) ~= numBinsOneSided
         warning('Noise spectrum size (%d) does not match expected one-sided size (%d). Adjusting.', size(noisePowerSpectrum,1), numBinsOneSided);
         % Attempt to resize (e.g., take first part or pad) - may indicate upstream error
         if size(noisePowerSpectrum, 1) > numBinsOneSided
             noisePowerSpectrum = noisePowerSpectrum(1:numBinsOneSided);
         else
             noisePowerSpectrum = [noisePowerSpectrum; zeros(numBinsOneSided-size(noisePowerSpectrum,1),1)];
         end
    end
    % --- End Check ---


    % Perform STFT
    if exist('stft', 'file') && exist('istft', 'file')
        fprintf('Using built-in stft/istft for Wiener filter.\n');
        [S, F, T] = stft(audio, fs, 'Window', window, 'OverlapLength', overlapSamples, 'FFTLength', fftSize);

        % --- Defensive Check ---
        if size(S, 1) ~= numBinsOneSided
             error('Built-in STFT returned unexpected number of frequency bins (%d vs expected %d). Check MATLAB version compatibility or signal type.', size(S,1), numBinsOneSided);
        end
        % --- End Check ---

        noisePowerMat = repmat(noisePowerSpectrum, 1, size(S, 2)); % Match dimensions

        % Wiener Filter Calculation (Power Spectrum Domain)
        signalPowerSpectrum = abs(S).^2;
        % A priori SNR estimation (Decision-Directed)
        alpha_snr = 0.98;
        snr_post = signalPowerSpectrum ./ (noisePowerMat + eps);
        snr_prio = zeros(size(snr_post));
        % Initialize first frame (handle potential zero noise)
        snr_prio(:,1) = max(snr_post(:,1) - 1, 0);
        for t = 2:size(S, 2)
            % Gain from previous estimate's SNR_prio
            gain_prev_sq = (snr_prio(:,t-1) ./ (snr_prio(:,t-1) + 1)).^2; % Squared gain H^2 = (SNRprio/(SNRprio+1))^2
            snr_prio(:,t) = alpha_snr * (gain_prev_sq .* snr_post(:,t-1)) + (1-alpha_snr) * max(snr_post(:,t) - 1, 0);
         end
         gain = snr_prio ./ (snr_prio + 1);

        % Apply gain and perform ISTFT
        S_enhanced = S .* gain;
        enhanced = istft(S_enhanced, fs, 'Window', window, 'OverlapLength', overlapSamples, 'FFTLength', fftSize, 'Method', 'ola');

    else
        % Manual STFT/ISTFT if functions not available
        warning('Built-in stft/istft not found or version mismatch. Using manual implementation for Wiener (less optimal).');
        numFrames = floor((len_in - overlapSamples) / hopSize);
        % Pre-allocate output slightly longer for OLA, will trim later
        olaBufferLength = len_in + fftSize;
        enhanced = zeros(olaBufferLength, 1);
        normWindowSum = zeros(olaBufferLength, 1);

        for i = 1:numFrames
            startIdx = (i-1) * hopSize + 1;
            endIdx = startIdx + frameSizeSamples - 1;
            if endIdx > len_in % Handle last frame
                frame = [audio(startIdx:len_in); zeros(endIdx-len_in, 1)];
            else
                frame = audio(startIdx:endIdx);
            end

            windowedFrame = frame .* window;
            frameFFT_full = fft(windowedFrame, fftSize);

            % --- MODIFICATION START: Work with one-sided ---
            frameFFT_oneSided = frameFFT_full(1:numBinsOneSided);
            framePowerSpectrum_oneSided = abs(frameFFT_oneSided).^2;
            % Optional: Scale power like pwelch/stft
            if fftSize > 2
                 framePowerSpectrum_oneSided(2:end-1) = framePowerSpectrum_oneSided(2:end-1); % No need to double here as we use full power noise est.
            end

            % Simple Wiener gain (using posteriori SNR on one-sided spectra)
            snr_post = framePowerSpectrum_oneSided ./ (noisePowerSpectrum + eps);
            gain = max(snr_post - 1, 0) ./ (snr_post + 1); % Gain = SNR_prio / (SNR_prio + 1) approx SNR_post / (SNR_post + 1)
            % gain = max(snr_post - 1, 0) ./ snr_post; % Alternative formulation G = max(SNRpost-1,0)/SNRpost

            % Apply gain to one-sided spectrum
            enhancedFrameFFT_oneSided = frameFFT_oneSided .* gain;

            % Reconstruct two-sided spectrum for IFFT
            if mod(fftSize, 2) == 0 % Even fftSize, Nyquist exists
                enhancedFrameFFT_full = [enhancedFrameFFT_oneSided; conj(enhancedFrameFFT_oneSided(end-1:-1:2))];
            else % Odd fftSize, no Nyquist
                enhancedFrameFFT_full = [enhancedFrameFFT_oneSided; conj(enhancedFrameFFT_oneSided(end:-1:2))];
            end
            % --- MODIFICATION END ---

            enhancedFrame = real(ifft(enhancedFrameFFT_full, fftSize)); % IFFT on reconstructed full spectrum

            % Overlap-add
            outStartIdx = startIdx;
            outEndIdx = outStartIdx + fftSize - 1; % Use fftSize for IFFT length

            % Ensure indices are within the allocated buffer
            if outEndIdx <= olaBufferLength
                 enhanced(outStartIdx:outEndIdx) = enhanced(outStartIdx:outEndIdx) + enhancedFrame;
                 % Accumulate window for normalization (use window, not squared, for OLA)
                 paddedWindow = [window; zeros(fftSize-frameSizeSamples,1)]; % Pad window to fftSize
                 normWindowSum(outStartIdx:outEndIdx) = normWindowSum(outStartIdx:outEndIdx) + paddedWindow;
            else
                 warning('OLA indices exceed buffer length in manual Wiener. Skipping frame %d addition.', i);
            end
       end

        % Normalize for window and overlap
        normWindowSum(normWindowSum < 1e-6) = 1; % Avoid division by zero
        enhanced = enhanced ./ normWindowSum;
        % Trim to original length BEFORE returning from this function
        enhanced = enhanced(1:len_in);

    end

    % Final length adjustment (redundant if manual path trims, but safe)
    enhanced = adjustSignalLength(enhanced, len_in);
end

function noisePowerSpectrum = estimateNoisePowerSpectrum(noiseEstimate, frameSize, window, fftSize)
    % Estimate noise power spectrum using Welch's method for stability
    if length(noiseEstimate) < frameSize
        % (Keep warning and repeating logic as is)
        noiseEstimate = repmat(noiseEstimate, ceil(frameSize/length(noiseEstimate)), 1);
        noiseEstimate = noiseEstimate(1:max(frameSize, length(noiseEstimate))); % Ensure min length
    end

    try
        % Use pwelch for a more stable estimate (already returns one-sided)
        [pxx, ~] = pwelch(noiseEstimate, window, floor(frameSize/2), fftSize);
        noisePowerSpectrum = pxx * (sum(window.^2)/frameSize); % Scale pwelch output to Power Spectrum
        noisePowerSpectrum = noisePowerSpectrum(:); % Ensure column vector
    catch ME
        warning('pwelch failed for noise estimation: %s. Using simpler average FFT.', ME.message);
        % Fallback: Average FFT over frames (less stable)
        hopSize = floor(frameSize / 2);
        numFrames = floor((length(noiseEstimate) - frameSize) / hopSize) + 1;
        if numFrames < 1; numFrames = 1; end % Handle short noise estimate

        noisePowerSpectrumSum = zeros(fftSize, 1); % Calculate full FFT first
        frameCount = 0;
        for i = 1:numFrames
            startIdx = (i-1)*hopSize + 1;
            endIdx = startIdx + frameSize - 1;
            if endIdx > length(noiseEstimate) % Handle last frame
                frame = [noiseEstimate(startIdx:end); zeros(endIdx-length(noiseEstimate),1)];
            else
                frame = noiseEstimate(startIdx:endIdx);
            end

            if length(frame) == frameSize
                windowedFrame = frame .* window;
                frameFFT = fft(windowedFrame, fftSize);
                noisePowerSpectrumSum = noisePowerSpectrumSum + abs(frameFFT).^2; % Sum power
                frameCount = frameCount + 1;
            end
        end

        if frameCount > 0
            noisePowerSpectrumFull = noisePowerSpectrumSum / frameCount; % Average full power spectrum
            % --- MODIFICATION START ---
            % Take only the first half (one-sided spectrum)
            numBinsOneSided = floor(fftSize/2) + 1;
            noisePowerSpectrum = noisePowerSpectrumFull(1:numBinsOneSided);
            % Optional: Double power for non-DC/Nyquist bins (consistent with pwelch scaling)
            if fftSize > 2
                 noisePowerSpectrum(2:end-1) = 2 * noisePowerSpectrum(2:end-1);
            end
             % --- MODIFICATION END ---
        else
            warning('Could not process any noise frames for PSD estimate. Returning zeros.');
            numBinsOneSided = floor(fftSize/2) + 1; % Calculate expected size
            noisePowerSpectrum = zeros(numBinsOneSided, 1); % Return zeros of correct size
        end
        noisePowerSpectrum = noisePowerSpectrum(:); % Ensure column
    end

    % Ensure non-negative and handle potential NaN/Inf
    noisePowerSpectrum(isnan(noisePowerSpectrum) | isinf(noisePowerSpectrum)) = 0;
    noisePowerSpectrum = max(noisePowerSpectrum, 0);
end

function enhanced = applyKalmanFilter(audio, config)
    len_in = length(audio);
    % Kalman filter for signal tracking (assuming signal is the state)
    % Simpler 1st order model might be sufficient unless tracking derivatives
    A = 1; % State transition matrix (assume signal is slowly varying)
    H = 1; % Observation matrix (direct observation)

    % Process and measurement noise
    Q = config.Q; % Process noise variance (how much signal is expected to change)
    R = config.R; % Measurement noise variance (how noisy the observation is)

    % Initialize state and covariance
    state = config.initial_state; % Initial guess of the signal value
    covar = config.initial_covar; % Initial uncertainty

    enhanced = zeros(size(audio));

    % Apply Kalman filter sample by sample
    for i = 1:len_in
        % Prediction step
        state_pred = A * state;
        covar_pred = A * covar * A' + Q;

        % Update step (Kalman gain calculation)
        K = covar_pred * H' / (H * covar_pred * H' + R);

        % Correct state estimate using measurement
        state = state_pred + K * (audio(i) - H * state_pred);

        % Update covariance
        covar = (1 - K * H) * covar_pred;

        % Output filtered estimate
        enhanced(i) = state;
    end

    % Ensure output length matches input length (though it should by design here)
    enhanced = adjustSignalLength(enhanced, len_in);
end

function enhanced = applySpectralSubtraction(audio, fs, noiseEstimate, config)
    len_in = length(audio);
    win_size = 1024; % Consider making this adaptive or configurable
    overlap_size = win_size / 2;
    hop_size = win_size - overlap_size;
    fft_size = win_size; % FFT size same as window size
    window = hamming(win_size, 'periodic');

    % --- Noise Spectrum Estimation ---
    noise_mag_spectrum = estimateNoiseMagnitudeSpectrum(noiseEstimate, win_size, window, fft_size);

    % --- STFT of input signal ---
    if exist('stft', 'file') && exist('istft', 'file')
        S = stft(audio, fs, 'Window', window, 'OverlapLength', overlap_size, 'FFTLength', fft_size);
    else
        warning('Spectral Subtraction: Built-in stft/istft not found. Manual STFT used (less optimal).');
        numFrames = floor((len_in - overlap_size) / hop_size);
        S = zeros(fft_size, numFrames); % Preallocate
         for i = 1:numFrames
             startIdx = (i-1) * hop_size + 1;
             endIdx = startIdx + win_size - 1;
             if endIdx > len_in
                 frame = [audio(startIdx:len_in); zeros(endIdx-len_in, 1)];
             else
                 frame = audio(startIdx:endIdx);
             end
             S(:, i) = fft(frame .* window, fft_size);
         end
    end

    % --- Spectral Subtraction Core ---
    mag = abs(S);
    phase = angle(S);

    % Parameters
    alpha = config.alpha; % Over-subtraction factor
    beta = config.beta;   % Spectral floor factor

    % Expand noise spectrum to match signal spectrogram dimensions
    noise_mag_mat = repmat(noise_mag_spectrum, 1, size(mag, 2));

    % Perform subtraction
    subtracted_mag = mag - alpha * noise_mag_mat;

    % Apply spectral floor
    spectral_floor = beta * noise_mag_mat;
    subtracted_mag = max(subtracted_mag, spectral_floor);

    % --- ISTFT ---
    S_enhanced = subtracted_mag .* exp(1i * phase);

    if exist('istft', 'file')
         enhanced = istft(S_enhanced, fs, 'Window', window, 'OverlapLength', overlap_size, 'FFTLength', fft_size, 'Method', 'ola');
    else
         warning('Spectral Subtraction: Built-in istft not found. Manual ISTFT used (less optimal).');
         enhanced = basicISTFT(S_enhanced, window, hop_size, len_in); % Use basic ISTFT helper
    end

    % Ensure output length matches input length
    enhanced = adjustSignalLength(enhanced, len_in);
end

function noise_mag_spectrum = estimateNoiseMagnitudeSpectrum(noiseEstimate, win_size, window, fft_size)
    % Similar to power spectrum estimation, but returns magnitude
    if length(noiseEstimate) < win_size
        warning('Noise estimate shorter than window size. Repeating for magnitude spectrum.');
        noiseEstimate = repmat(noiseEstimate, ceil(win_size/length(noiseEstimate)), 1);
        noiseEstimate = noiseEstimate(1:max(win_size, length(noiseEstimate)));
    end

    try
        % Use pwelch and take sqrt for magnitude
        [pxx, ~] = pwelch(noiseEstimate, window, floor(win_size/2), fft_size);
        noise_power_spectrum = pxx * (sum(window.^2)/win_size); % Scale to Power Spectrum
        noise_mag_spectrum = sqrt(max(noise_power_spectrum, 0)); % Magnitude is sqrt(Power)
        noise_mag_spectrum = noise_mag_spectrum(:); % Ensure column
    catch ME
        warning('pwelch failed for noise magnitude estimation: %s. Using simpler average FFT magnitude.', ME.message);
        % Fallback: Average FFT magnitude
        hop_size = floor(win_size / 2);
        numFrames = floor((length(noiseEstimate) - win_size) / hop_size) + 1;
        if numFrames < 1; numFrames = 1; end

        noise_mag_spectrum_sum = zeros(fft_size, 1);
        frameCount = 0;
        for i = 1:numFrames
            startIdx = (i-1)*hop_size + 1;
            endIdx = startIdx + win_size - 1;
             if endIdx > length(noiseEstimate)
                 frame = [noiseEstimate(startIdx:end); zeros(endIdx-length(noiseEstimate),1)];
             else
                 frame = noiseEstimate(startIdx:endIdx);
             end
             if length(frame) == win_size
                frameFFT = fft(frame .* window, fft_size);
                noise_mag_spectrum_sum = noise_mag_spectrum_sum + abs(frameFFT);
                frameCount = frameCount + 1;
             end
        end
         if frameCount > 0
            noise_mag_spectrum = noise_mag_spectrum_sum / frameCount;
         else
             warning('Could not process noise frames for magnitude estimate. Returning zeros.');
             noise_mag_spectrum = zeros(fft_size, 1);
         end
    end
     % Ensure non-negative and handle potential NaN/Inf
     noise_mag_spectrum(isnan(noise_mag_spectrum) | isinf(noise_mag_spectrum)) = 0;
end


function enhanced = basicISTFT(S_enhanced, window, hop_size, original_length)
    % Basic ISTFT implementation using overlap-add
    [fft_size, num_frames] = size(S_enhanced);
    win_size = length(window); % Get actual window length

    % Calculate expected output length from OLA parameters
    output_length = hop_size * (num_frames - 1) + win_size;
    enhanced = zeros(output_length, 1);
    window_sum = zeros(output_length, 1); % For normalization

    for i = 1:num_frames
        % IFFT of current frame
        frame = real(ifft(S_enhanced(:, i), fft_size)); % Use fft_size
        frame = frame(1:win_size); % Truncate to window size

        % Overlap-add
        startIdx = (i-1) * hop_size + 1;
        endIdx = startIdx + win_size - 1;

        enhanced(startIdx:endIdx) = enhanced(startIdx:endIdx) + frame .* window;
        window_sum(startIdx:endIdx) = window_sum(startIdx:endIdx) + window.^2; % Use window squared for OLA normalization
    end

    % Normalize based on window sum to correct amplitude
    window_sum(window_sum < 1e-6) = 1; % Avoid division by zero
    enhanced = enhanced ./ window_sum;

    % Adjust to the *specified* original_length
    enhanced = adjustSignalLength(enhanced, original_length);
end

function enhanced = applyMedianFilter(audio)
    len_in = length(audio);
    % Apply median filtering with a fixed small window size
    windowSize = 5; % Typical size for removing clicks/spikes

    try
        enhanced = medfilt1(audio, windowSize, 'truncate'); % 'truncate' avoids padding issues affecting length
    catch ME
        warning('Median filter failed: %s. Returning original audio.', ME.message);
        enhanced = audio;
    end

    % Ensure output length matches input length
    enhanced = adjustSignalLength(enhanced, len_in);
end

function enhanced = applyWaveletDenoising(audio, config)
    len_in = length(audio);
    reconstruction_successful = false; % Flag

    try
        % Decompose the signal
        [C, L] = wavedec(audio, config.level, config.family);

        % Estimate noise level robustly (using MAD of finest detail)
        try
            detail1 = detcoef(C, L, 1);
            if isempty(detail1)
                warning('Wavelet: Level 1 detail coefficients are empty. Using global MAD.');
                sigma = median(abs(audio - median(audio))) / 0.6745; % More robust MAD
            else
                sigma = median(abs(detail1 - median(detail1))) / 0.6745;
            end
        catch ME_detail
            warning('Wavelet: Error extracting level 1 details (%s). Using global MAD.', ME_detail.message);
            sigma = median(abs(audio - median(audio))) / 0.6745;
        end

        % Ensure sigma is a valid positive number
        if ~isfinite(sigma) || sigma <= 1e-9 % Check against small threshold
           warning('Wavelet: Invalid sigma calculated (%.4g). Using a small default.', sigma);
           sigma = 1e-6; % Use a small positive value if calculation failed
        end
        fprintf('Wavelet estimated noise sigma: %.4g\n', sigma);

        % Universal Threshold
        threshold = sigma * sqrt(2 * log(len_in));

        % Apply thresholding level by level
        C_denoised = C; % Work on a copy
        for i = 1:config.level % Iterate through detail levels 1 to N
            level_idx_in_L = config.level + 2 - i; % Index in L for level i detail length

            % --- Corrected Index Calculation in C ---
            % Start index: sum of lengths of all preceding coefficients + 1
            % Preceding coeffs: App(N), Det(N), Det(N-1), ..., Det(i+1)
            % Indices in L for these lengths: 1, 2, ..., (level + 1 - i)
            d_start = sum(L(1 : config.level + 1 - i)) + 1;
            % Length of detail coeffs for level i
            level_length = L(level_idx_in_L);
            % End index
            d_end = d_start + level_length - 1;

            % Basic sanity check for indices
            if d_start < 1 || d_end > length(C) || level_length < 0
                 warning('Wavelet: Invalid index calculation for level %d (start=%d, end=%d, len=%d). Skipping level.', i, d_start, d_end, level_length);
                 continue; % Skip this level
            end

            % Get detail coefficients for level 'i'
            detail_coeffs = C(d_start:d_end);

            % Apply thresholding (soft or hard)
            level_threshold = threshold; % Can adjust per level if desired: threshold * (some_factor_based_on_i)
            denoised_detail = wthresh(detail_coeffs, config.threshold_type, level_threshold);

            % Place denoised coefficients back into the vector
            C_denoised(d_start:d_end) = denoised_detail;
        end

        % Reconstruct signal
        try
            enhanced = waverec(C_denoised, L, config.family);
            reconstruction_successful = true;
            fprintf('Wavelet reconstruction successful.\n');
        catch ME_rec
            warning('Wavelet reconstruction failed: %s. Returning original audio for wavelet step.', ME_rec.message);
            enhanced = audio; % Fallback if reconstruction fails
        end

    catch ME_wavelet
        fprintf('Error during wavelet processing: %s\n', ME_wavelet.message);
        fprintf('Wavelet error details: %s\n', getReport(ME_wavelet, 'extended', 'hyperlinks','off'));
        enhanced = audio; % Fallback to original audio if any wavelet step fails
    end

    % Ensure output length matches input length
    enhanced = adjustSignalLength(enhanced, len_in);

end

function weights = calculateAdaptiveWeights(audio, noiseEstimate, fs, config)
    % Calculate weights based on estimated SNR in different bands
    weights = struct('wiener', 0, 'kalman', 0, 'spectral', 0, 'median', 0, 'wavelet', 0); % Initialize
    numEnabled = config.useWiener + config.useKalman + config.useSpectral + config.useMedian + config.useWavelet;
    defaultWeight = ternary(numEnabled > 0, 1/numEnabled, 0);

    try
        win_len = min([length(audio), length(noiseEstimate), 1024]); % Window for pwelch
        if win_len < 2
             error('Signal too short for PSD estimation.');
        end

        [psd_signal, f] = pwelch(audio, hamming(win_len), [], [], fs);
        [psd_noise, ~]  = pwelch(noiseEstimate, hamming(win_len), [], [], fs);

        % Ensure noise PSD is not zero to avoid Inf SNR
        psd_noise = max(psd_noise, eps);

        % Calculate SNR in dB for different frequency bands
        f_low = 500; f_mid = 2000;
        snr_low  = 10*log10(mean(psd_signal(f < f_low)) / mean(psd_noise(f < f_low)));
        snr_mid  = 10*log10(mean(psd_signal(f >= f_low & f < f_mid)) / mean(psd_noise(f >= f_low & f < f_mid)));
        snr_high = 10*log10(mean(psd_signal(f >= f_mid)) / mean(psd_noise(f >= f_mid)));

        % Handle potential -Inf SNR if signal power is zero in a band
        snr_low = max(snr_low, -100); % Floor SNR
        snr_mid = max(snr_mid, -100);
        snr_high = max(snr_high, -100);

        % Weighted SNR across bands (emphasize mid-frequencies for speech)
        snr_weighted = 0.2 * snr_low + 0.5 * snr_mid + 0.3 * snr_high;
        fprintf('Adaptive Weights - SNR Estimate: Low=%.1fdB, Mid=%.1fdB, High=%.1fdB, Weighted=%.1fdB\n', ...
                snr_low, snr_mid, snr_high, snr_weighted);

        % Define weights based on SNR (Example logic, needs tuning)
        if snr_weighted > 12 % High SNR: Trust signal more, less aggressive filtering
            w_wiener = 0.4; w_kalman = 0.3; w_spectral = 0.0; w_median = 0.1; w_wavelet = 0.2;
        elseif snr_weighted > 3 % Medium SNR: Balanced approach
            w_wiener = 0.3; w_kalman = 0.2; w_spectral = 0.0; w_median = 0.2; w_wavelet = 0.3;
        else % Low SNR: Prioritize noise removal
            w_wiener = 0.2; w_kalman = 0.1; w_spectral = 0.0; w_median = 0.3; w_wavelet = 0.4;
        end

        % Apply calculated weights, respecting enabled flags
        weights.wiener   = config.useWiener * w_wiener;
        weights.kalman   = config.useKalman * w_kalman;
        weights.spectral = config.useSpectral * w_spectral;
        weights.median   = config.useMedian * w_median;
        weights.wavelet  = config.useWavelet * w_wavelet;

        % Normalize weights to sum to 1
        total_weight = sum(structfun(@(x) x, weights));
        if total_weight > 1e-6
            weights.wiener   = weights.wiener / total_weight;
            weights.kalman   = weights.kalman / total_weight;
            weights.spectral = weights.spectral / total_weight;
            weights.median   = weights.median / total_weight;
            weights.wavelet  = weights.wavelet / total_weight;
        else
            % Fallback if all weights ended up zero (e.g., all disabled)
             weights.wiener   = config.useWiener * defaultWeight;
             weights.kalman   = config.useKalman * defaultWeight;
             weights.spectral = config.useSpectral * defaultWeight;
             weights.median   = config.useMedian * defaultWeight;
             weights.wavelet  = config.useWavelet * defaultWeight;
        end

    catch ME_weights
        warning('Failed to calculate adaptive weights: %s. Using default equal weights.', ME_weights.message);
        weights.wiener   = config.useWiener * defaultWeight;
        weights.kalman   = config.useKalman * defaultWeight;
        weights.spectral = config.useSpectral * defaultWeight;
        weights.median   = config.useMedian * defaultWeight;
        weights.wavelet  = config.useWavelet * defaultWeight;
    end

    fprintf('Final Weights: Wiener=%.2f, Kalman=%.2f, Spectral=%.2f, Median=%.2f, Wavelet=%.2f\n', ...
        weights.wiener, weights.kalman, weights.spectral, weights.median, weights.wavelet);
end

function audio_out = applyPostProcessing(audio_in, fs)
    % Apply post-processing steps

    % 1. Gentle low-pass filter to smooth potential artifacts
    try
        cutoff = min(fs/2 - 500, 7500); % Cutoff around 7.5kHz or below Nyquist
if cutoff > 100 % Only apply if cutoff is reasonable
    [b, a] = butter(3, cutoff/(fs/2), 'low'); % 3rd order Butterworth
    audio_out = filter(b, a, audio_in);
    fprintf('Applied post-processing LPF (Cutoff: %.0f Hz).\n', cutoff);
        else
             audio_out = audio_in; % Skip if cutoff too low
        end
    catch ME
        warning('Post-processing LPF failed: %s', ME.message);
        audio_out = audio_in;
    end


    % 2. Optional: Very light dynamic range compression or peak limiting
    % Simple peak limiting example:
    threshold = 0.98; % Limit peaks slightly below max
    audio_out(audio_out > threshold) = threshold;
    audio_out(audio_out < -threshold) = -threshold;
    % Note: A proper compressor (e.g., using audioCompressor object) is better
    % but adds complexity and toolbox dependency.

    % Ensure length hasn't changed (filter might add negligible delay)
    audio_out = adjustSignalLength(audio_out, length(audio_in));
end

function enhanced = applyBasicEnhancement(audio, fs)
    % Fallback basic enhancement (simple filtering)
    fprintf('Applying basic fallback enhancement (HPF).\n');
    len_in = length(audio);
    enhanced = audio; % Start with original

    try
        % Apply simple high-pass filter
        cutoff = 100; % Fixed 100 Hz cutoff
        if fs/2 > cutoff
             [b, a] = butter(2, cutoff/(fs/2), 'high');
             enhanced = filter(b, a, audio);
        end
    catch ME
        warning('Basic enhancement HPF failed: %s', ME.message);
        enhanced = audio; % Revert to original if filter fails
    end

    % Normalize (simple max normalization)
     peak = max(abs(enhanced(:)));
     if peak > 1e-9
         enhanced = enhanced / peak * 0.95; % Normalize to 0.95 peak
     end

    % Ensure output length matches input length
    enhanced = adjustSignalLength(enhanced, len_in);
end

function signal_out = adjustSignalLength(signal_in, target_length)
    % Utility to pad or truncate a signal to a target length
    current_length = length(signal_in);
    if current_length == target_length
        signal_out = signal_in;
    elseif current_length > target_length
        signal_out = signal_in(1:target_length); % Truncate
    else % current_length < target_length
        padding = zeros(target_length - current_length, 1);
        signal_out = [signal_in(:); padding]; % Pad with zeros (ensure column)
    end
end

% =========================================================================
% Visualization Stage
% =========================================================================

function generateVisualizations(originalAudio, processedAudio, enhancedAudio, originalFs, fs, visualizationPrefix, preprocessInfo)
    fprintf('Generating visualizations...\n');

    % Ensure consistent lengths for plotting where necessary
    len_orig = length(originalAudio);
    len_proc = length(processedAudio);
    len_enh = length(enhancedAudio);

    % --- Time Domain Plot ---
    try
        figure('visible','off', 'Name', 'Time Domain Comparison');
        t_orig = (0:len_orig-1) / originalFs;
        t_proc = (0:len_proc-1) / fs; % Use current fs for processed/enhanced
        t_enh = (0:len_enh-1) / fs;

        subplot(3, 1, 1);
        plot(t_orig, originalAudio);
        title('Original Audio'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 max([t_orig, t_proc, t_enh])]);

        subplot(3, 1, 2);
        plot(t_proc, processedAudio);
        title('Pre-Processed Audio'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 max([t_orig, t_proc, t_enh])]);

        subplot(3, 1, 3);
        plot(t_enh, enhancedAudio);
        title('Enhanced Audio'); xlabel('Time (s)'); ylabel('Amplitude'); grid on; xlim([0 max([t_orig, t_proc, t_enh])]);

        saveas(gcf, [visualizationPrefix '_1_time_domain.png']);
        close(gcf);
    catch ME_plot
        warning('Failed to generate time domain plot: %s', ME_plot.message);
        if exist('gcf', 'var') && ishandle(gcf); close(gcf); end % Close figure if open
    end

    % --- Spectrogram Plot ---
    try
        figure('visible','off', 'Name', 'Spectrogram Comparison');
        win_spec = min(512, floor(len_orig/4)); % Adjust window based on signal length
        overlap_spec = floor(win_spec/2);
        fft_spec = win_spec;

        subplot(3, 1, 1);
        if len_orig > win_spec % Check if long enough for spectrogram
            spectrogram(originalAudio, hamming(win_spec), overlap_spec, fft_spec, originalFs, 'yaxis');
            title('Original Audio Spectrogram'); colorbar('off');
        else
            title('Original Audio (Too short for spectrogram)'); axis off;
        end


        subplot(3, 1, 2);
         win_spec_proc = min(512, floor(len_proc/4)); overlap_spec_proc=floor(win_spec_proc/2);fft_spec_proc=win_spec_proc;
        if len_proc > win_spec_proc
            spectrogram(processedAudio, hamming(win_spec_proc), overlap_spec_proc, fft_spec_proc, fs, 'yaxis');
            title('Pre-Processed Audio Spectrogram'); colorbar('off');
        else
             title('Pre-Processed Audio (Too short for spectrogram)'); axis off;
        end

        subplot(3, 1, 3);
         win_spec_enh = min(512, floor(len_enh/4)); overlap_spec_enh=floor(win_spec_enh/2);fft_spec_enh=win_spec_enh;
        if len_enh > win_spec_enh
            spectrogram(enhancedAudio, hamming(win_spec_enh), overlap_spec_enh, fft_spec_enh, fs, 'yaxis');
            title('Enhanced Audio Spectrogram'); colorbar('off');
        else
             title('Enhanced Audio (Too short for spectrogram)'); axis off;
        end

        saveas(gcf, [visualizationPrefix '_2_spectrogram.png']);
        close(gcf);
    catch ME_plot
        warning('Failed to generate spectrogram plot: %s', ME_plot.message);
         if exist('gcf', 'var') && ishandle(gcf); close(gcf); end
    end

    % --- Framing Visualization ---
    try
        if isfield(preprocessInfo, 'framedSegments') && ~isempty(preprocessInfo.framedSegments) && iscell(preprocessInfo.framedSegments)
             numFramesVis = length(preprocessInfo.framedSegments);
             if numFramesVis > 0
                figure('visible','off', 'Name', 'Framing Example');
                maxSubplots = min(numFramesVis, 5); % Limit subplots
                for i = 1:maxSubplots
                    subplot(maxSubplots, 1, i);
                     if ~isempty(preprocessInfo.framedSegments{i})
                         frameData = preprocessInfo.framedSegments{i};
                         frameLength = length(frameData);
                         timeAxis = (0:frameLength-1) / fs; % Use current fs
                         plot(timeAxis, frameData);
                         title(['Framed/Windowed Segment #' num2str(i)]); xlabel('Time (s)'); ylabel('Amplitude'); grid on;
                     else
                          title(['Framed/Windowed Segment #' num2str(i) ' (Empty)']); axis off;
                     end
                end
                saveas(gcf, [visualizationPrefix '_3_framing.png']);
                close(gcf);
             end
        end
    catch ME_plot
        warning('Failed to generate framing plot: %s', ME_plot.message);
         if exist('gcf', 'var') && ishandle(gcf); close(gcf); end
    end

    % --- Filter Response Visualization ---
    try
        if isfield(preprocessInfo, 'filterCoeff') && ~isempty(preprocessInfo.filterCoeff)
            figure('visible','off', 'Name', 'HPF Response');
            freqz(preprocessInfo.filterCoeff.b, preprocessInfo.filterCoeff.a, 1024, fs);
            title('High-Pass Filter Frequency Response');
            saveas(gcf, [visualizationPrefix '_4_highpass_response.png']);
            close(gcf);
        end
    catch ME_plot
        warning('Failed to generate filter response plot: %s', ME_plot.message);
         if exist('gcf', 'var') && ishandle(gcf); close(gcf); end
    end

    % --- PSD Plot ---
    try
        generatePSDPlot(originalAudio, enhancedAudio, originalFs, fs, visualizationPrefix);
    catch ME_plot
        warning('Failed to generate PSD plot: %s', ME_plot.message);
         if exist('gcf', 'var') && ishandle(gcf); close(gcf); end
    end

    fprintf('Visualizations generated for %s\n', visualizationPrefix);
end

function generatePSDPlot(originalAudio, enhancedAudio, originalFs, fs, visualizationPrefix)
    % Generate and save PSD plot comparing original and enhanced
    figure('visible','off', 'Name', 'PSD Comparison');

    % Ensure signals are long enough for pwelch analysis
    win_psd = 1024; % Welch window size
    if length(originalAudio) < win_psd || length(enhancedAudio) < win_psd
         warning('Signals too short for reliable PSD estimation with window size %d. Skipping PSD plot.', win_psd);
         close(gcf);
         return;
    end

    [psdOriginal, fOriginal] = pwelch(originalAudio, hamming(win_psd), win_psd/2, win_psd, originalFs);
    [psdEnhanced, fEnhanced] = pwelch(enhancedAudio, hamming(win_psd), win_psd/2, win_psd, fs); % Use current fs for enhanced

    plot(fOriginal, 10*log10(psdOriginal + eps), 'b', 'LineWidth', 1); % Add eps for log stability
    hold on;
    plot(fEnhanced, 10*log10(psdEnhanced + eps), 'r', 'LineWidth', 1);
    hold off;

    title('Power Spectral Density (PSD)');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    legend('Original', 'Enhanced', 'Location', 'southwest');
    grid on;
    xlim([0 max(originalFs/2, fs/2)]); % Show up to highest Nyquist freq
    ylim_curr = ylim;
    ylim([max(ylim_curr(1), -100), ylim_curr(2)+5]); % Adjust y-axis, floor at -100dB

    saveas(gcf, [visualizationPrefix '_5_psd.png']);
    close(gcf);
end
