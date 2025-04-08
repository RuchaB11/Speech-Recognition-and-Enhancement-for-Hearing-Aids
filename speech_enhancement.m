clc;
close all;
clear all;

% Ensure required toolboxes are installed: Signal Processing, Audio, Control Systems

% Set input and output file paths
inputFile = 'C:\Users\sathw\Documents\MATLAB\testing2.wav';  % Change to your input WAV file
outputFile = 'C:\Users\sathw\Documents\MATLAB\enhanced_speech2.wav'; % Output enhanced WAV file

% Parameters (TUNING REQUIRED - CRITICAL FOR OPTIMAL PERFORMANCE)
frame_length = 0.025;       % Frame length in seconds (slightly longer for better speech context)
frame_overlap_fraction = 0.75; % Increased overlap for smoother transitions
preemphasis_coeff = 0.97;   % Pre-emphasis filter coefficient
targetNoiseFloorDB = -70;   % Lower target noise floor for more aggressive noise reduction

% Kalman Filter Parameters (TUNING REQUIRED)
Q = 1e-6;                   % Reduced process noise for stable tracking of speech
R = 0.01;                   % Reduced measurement noise for trusting input signal more
initial_state_estimate = 0;  % Initial state estimate
initial_error_covariance = 0.1;% Reduced initial error covariance

% Spectral Subtraction Parameters (TUNING REQUIRED)
alpha_sub = 3;          % Increased over-subtraction for more music removal
beta_sub = 0.001;       % Lower spectral floor to reduce artifacts when music is removed

% Wavelet Denoising Parameters (TUNING REQUIRED)
wavelet_level = 6;      % Number of wavelet decomposition levels
wavelet_family = 'db4';   % Wavelet family (Daubechies 4 is a good compromise)
threshold_type = 's'; % Threshold type ('s' for soft, 'h' for hard) - corrected to be 's' or 'h'

% Read input audio
try
    [noisyAudio, fs] = audioread(inputFile);
    if isempty(noisyAudio)
        error('Audio file is empty.');
    end
    
    % Convert floating-point PCM to standard PCM if necessary
    if ~isa(noisyAudio, 'double')
        noisyAudio = double(noisyAudio);
    end

    % Make sure audio is mono. If stereo, convert to mono.
    if size(noisyAudio, 2) > 1
        noisyAudio = mean(noisyAudio, 2);
        fprintf('Input audio was stereo. Converting to mono.\n');
    end
    
catch ME
    error('Error reading input WAV file: %s', ME.message);
end

% Apply pre-emphasis filter
noisyAudio = preemphasis(noisyAudio, preemphasis_coeff);

% Enhance audio using Wiener, Kalman filters, and additional methods
try
    [enhancedSpeech] = enhanceAudio(noisyAudio, fs, frame_length, frame_overlap_fraction, targetNoiseFloorDB, Q, R, initial_state_estimate, initial_error_covariance, alpha_sub, beta_sub, wavelet_level, wavelet_family, threshold_type);
    
    % Normalize audio to prevent clipping
    enhancedSpeech = enhancedSpeech / max(abs(enhancedSpeech));
    
    % Write output WAV file
    audiowrite(outputFile, enhancedSpeech, fs, 'BitsPerSample', 16);
    fprintf('Enhanced audio saved to: %s\n', outputFile);
catch ME
    error('Error during enhancement: %s', ME.message);
end

%% Helper Functions
function [enhancedSpeech] = enhanceAudio(noisyAudio, fs, frame_length, frame_overlap_fraction, targetNoiseFloorDB, Q, R, initial_state_estimate, initial_error_covariance, alpha_sub, beta_sub, wavelet_level, wavelet_family, threshold_type)
    noiseFloorDB = 10*log10(mean(noisyAudio.^2) + eps);
    gainAdjustDB = targetNoiseFloorDB - noiseFloorDB;
    gainAdjustLinear = 10.^(gainAdjustDB/20);
    cleanAudioEstimate = noisyAudio * gainAdjustLinear;
    cleanAudioEstimate = max(min(cleanAudioEstimate, 1), -1);
    cleanAudioEstimate = cast(cleanAudioEstimate, class(noisyAudio));

    % Apply all enhancement techniques
    enhancedSpeechWiener = wienerFilterEnhancement(noisyAudio, cleanAudioEstimate, fs, frame_length, frame_overlap_fraction);
    enhancedSpeechKalman = kalmanFilterEnhancement(noisyAudio, Q, R, initial_state_estimate, initial_error_covariance);
    enhancedSpeechSpectralSubtraction = zeros(size(noisyAudio)); %Skip the enhancedSpeechSpectralSubtraction call;
    enhancedSpeechMedianFilter = medianFilterEnhancement(noisyAudio);
    enhancedSpeechWavelet = waveletDenoisingEnhancement(noisyAudio, wavelet_level, wavelet_family, threshold_type);
    
    % Ensure all enhanced signals have the same length
    minLength = min([length(enhancedSpeechWiener), length(enhancedSpeechKalman), length(enhancedSpeechSpectralSubtraction), length(enhancedSpeechMedianFilter), length(enhancedSpeechWavelet)]);
    
    % Trim all signals to the minimum length
    enhancedSpeechWiener = enhancedSpeechWiener(1:minLength);
    enhancedSpeechKalman = kalmanFilterEnhancement(noisyAudio, Q, R, initial_state_estimate, initial_error_covariance);
    enhancedSpeechSpectralSubtraction = enhancedSpeechSpectralSubtraction(1:minLength);
    enhancedSpeechMedianFilter = medianFilterEnhancement(noisyAudio);
    enhancedSpeechWavelet = waveletDenoisingEnhancement(noisyAudio, wavelet_level, wavelet_family, threshold_type);

     % Adaptive Combination Weights (Can be further improved with Voice Activity Detection - VAD)
    % This attempts to prioritize Wiener and Kalman when signal is strong,
    % Spectral Subtraction when noise is dominant, and uses wavelet/median as smoothing.

    % Calculate a simple SNR proxy (modify as needed)
    % Frequency-Domain SNR Proxy:
    win_size_snr = 256; % Smaller window size than the main spectrogram (TUNING REQUIRED)
    overlap_size_snr = win_size_snr / 2;
    [S_noisy, ~, ~] = spectrogram(noisyAudio, hann(win_size_snr), overlap_size_snr, win_size_snr, fs);
    [S_clean, ~, ~] = spectrogram(cleanAudioEstimate, hann(win_size_snr), overlap_size_snr, win_size_snr, fs);

    signal_power = mean(abs(S_clean).^2, 2); % Average power in each frequency bin
    noise_power = mean(abs(S_noisy - S_clean).^2, 2);
    snr_freq = 10*log10(signal_power ./ (noise_power + eps));

    % Use the median SNR across frequency bins as the proxy
    snr_proxy = median(snr_freq);


    % Define weighting factors based on SNR proxy. Adjust these ranges carefully. (TUNING REQUIRED)
    if snr_proxy > 5 % High SNR, prioritize Wiener and Kalman
        weight_wiener = 0.8;       % Boost Wiener to preserve speech
        weight_kalman = 0.15;       % Keep Kalman relatively high for speech
        weight_spectral = 0.0;   % Significantly reduce spectral subtraction
        weight_median = 0.025;
        weight_wavelet = 0.025;
    elseif snr_proxy > -5 % Medium SNR, balance the techniques
        weight_wiener = 0.6;       % Boost Wiener a bit
        weight_kalman = 0.3;       % Keep Kalman high
        weight_spectral = 0.0;      % Reduce spectral subtraction
        weight_median = 0.05;
        weight_wavelet = 0.05;
    else % Low SNR, rely more on Spectral Subtraction and Wavelet denoising
        weight_wiener = 0.3;
        weight_kalman = 0.2;
        weight_spectral = 0.0;      % Increased spectral subtraction
        weight_median = 0.25;
        weight_wavelet = 0.25;
    end

    % Normalize the weights to sum to 1
    weights_sum = weight_wiener + weight_kalman + weight_spectral + weight_median + weight_wavelet;
    weight_wiener = weight_wiener / weights_sum;
    weight_kalman = weight_kalman / weights_sum;
    weight_spectral = weight_spectral / weights_sum;
    weight_median = weight_median / weights_sum;
    weight_wavelet = weight_wavelet / weights_sum;
    
    % Combine Enhanced Signals (Adaptive Weighted Average)
    enhancedSpeech = (weight_wiener * enhancedSpeechWiener) + (weight_kalman * enhancedSpeechKalman) + ...
                     (weight_spectral * enhancedSpeechSpectralSubtraction) + (weight_median * enhancedSpeechMedianFilter) + ...
                     (weight_wavelet * enhancedSpeechWavelet);
end

function y = preemphasis(x, alpha)
    y = filter([1 -alpha], 1, x);
end

% Wiener Filter Enhancement
function enhancedSignal = wienerFilterEnhancement(noisySignal, cleanSignal, fs, frame_length, frame_overlap_fraction)
    frame_size = round(frame_length * fs);
    overlap_size = round(frame_size * frame_overlap_fraction);
    enhancedSignal = zeros(size(noisySignal));
    numFrames = floor((length(noisySignal) - frame_size) / (frame_size - overlap_size)) + 1;

    % Windowing function (e.g., Hamming)
    window = hamming(frame_size);  % Apply Hamming window

    for i = 1:numFrames
        startIdx = (i-1) * (frame_size - overlap_size) + 1;
        endIdx = min(startIdx + frame_size - 1, length(noisySignal)); % Handle edge case:  The fix is adding a MIN to ensure we can never exceed the length of signal

        % Zero-pad if necessary
        if (endIdx - startIdx + 1) < frame_size
            noisyFrame = [noisySignal(startIdx:endIdx); zeros(frame_size - (endIdx - startIdx + 1), 1)];
            cleanFrame = [cleanSignal(startIdx:endIdx); zeros(frame_size - (endIdx - startIdx + 1), 1)];

        else
            noisyFrame = noisySignal(startIdx:endIdx);
            cleanFrame = cleanSignal(startIdx:endIdx);
        end

        % Apply window
        noisyFrame = noisyFrame .* window;
        cleanFrame = cleanFrame .* window;


        noisePower = mean((noisyFrame - cleanFrame).^2) + eps;  % Noise power estimate.  Use (noisy - clean) for better estimate of the actual noise
        signalPower = mean(cleanFrame.^2) + eps;                  % Signal power estimate
        wienerGain = signalPower ./ (signalPower + noisePower);    % Wiener gain calculation (element-wise division)

        wienerGain = min(max(wienerGain, 0), 1); % Clip the gain to be between 0 and 1.

        enhancedFrame = wienerGain .* noisyFrame;  % Apply Wiener filter in time domain

        % Overlap-add
        frameLengthToCopy = length(enhancedSignal(startIdx:endIdx));
        enhancedSignal(startIdx:endIdx) = enhancedSignal(startIdx:endIdx) + enhancedFrame(1:frameLengthToCopy);  %ensure length matches
    end

    % Normalize
    enhancedSignal = enhancedSignal ./ max(abs(enhancedSignal));

end

% Kalman Filter Enhancement
function enhancedSignal = kalmanFilterEnhancement(noisySignal, Q, R, initial_state_estimate, initial_error_covariance)
    state_estimate = initial_state_estimate;
    error_covariance = initial_error_covariance;
    enhancedSignal = zeros(size(noisySignal));
    
    for n = 1:length(noisySignal)
        predicted_state = state_estimate;
        predicted_error_covariance = error_covariance + Q;
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + R);
        state_estimate = predicted_state + kalman_gain * (noisySignal(n) - predicted_state);
        error_covariance = (1 - kalman_gain) * predicted_error_covariance;
        enhancedSignal(n) = state_estimate;
    end
end

% Spectral Subtraction Enhancement
function enhancedSignal = spectralSubtractionEnhancement(noisySignal, fs, alpha_sub, beta_sub)
    win_size = 1024;
    overlap_size = win_size / 2;
    
    % Ensure noisySignal is a column vector
    noisySignal = noisySignal(:);

    [S, F, T] = spectrogram(noisySignal, hann(win_size), overlap_size, win_size, fs);  % Use Hann window
    magnitude = abs(S);
    phase = angle(S);

    % Improved Noise Estimation (Average of first few frames, assuming they are noise only)
    num_noise_frames = min(5, size(magnitude, 2));  % Use up to first 5 frames for noise estimation
    noise_estimate = mean(magnitude(:, 1:num_noise_frames), 2);

    % Spectral Subtraction with Over-Subtraction and Spectral Floor
    magnitude_subtracted = max(magnitude - alpha_sub * noise_estimate, beta_sub * noise_estimate);
    
    % Apply the phase of the noisy signal to the modified magnitude
    S_enhanced = magnitude_subtracted .* exp(1i * phase);

    % Replace the ISTFT section with:
    % Inverse Short-Time Fourier Transform (ISTFT)
    try
        % Use istft from the Signal Processing Toolbox correctly. Remove 'SampleRate'
        enhancedSignal = istft(S_enhanced, 'Window', hann(win_size), ...
                               'OverlapLength', overlap_size, ...
                               'FFTLength', win_size);
    catch ME
        % If that fails, try a manual implementation
        warning('ISTFT failed: %s. Trying manual reconstruction.', ME.message);
        try
            % Manual IFFT-based reconstruction
            y_rec = zeros((size(S_enhanced, 2)-1)*(win_size-overlap_size) + win_size, 1);
            w = hann(win_size);

            for i = 1:size(S_enhanced, 2)
                % Get current segment
                segment = ifft(S_enhanced(:, i), win_size, 'symmetric');
                segment = real(segment) .* w;

                % Add to output
                idx_start = (i-1)*(win_size-overlap_size) + 1;
                idx_end = min(idx_start + win_size - 1, length(noisySignal)); %prevent overindexing.
                segment = segment(1:idx_end - idx_start + 1); %ensure segments match
                y_rec(idx_start:idx_end) = y_rec(idx_start:idx_end) + segment;
            end

            enhancedSignal = y_rec;
        catch
            warning('Manual reconstruction also failed. Returning zero vector.');
            enhancedSignal = zeros(size(noisySignal));
            return;
        end
    end

    % Ensure the output is the same length as the input. Handle potential length mismatches.
    enhancedSignal = enhancedSignal(1:length(noisySignal));

end

% Median Filtering Enhancement
function enhancedSignal = medianFilterEnhancement(noisySignal)
    % Apply a median filter to reduce impulse noise.  Experiment with window size.
    enhancedSignal = medfilt1(noisySignal, 3); % Reduced window size.  Too large a window can smear the speech.
end

% Wavelet Denoising Enhancement
function enhancedSignal = waveletDenoisingEnhancement(noisySignal, wavelet_level, wavelet_family, threshold_type)

    [C, L] = wavedec(noisySignal, wavelet_level, wavelet_family);  % Decompose signal using specified wavelet

    % Improved Thresholding (VisuShrink or SureShrink may be even better)
    sigma = median(abs(C)) / 0.6745;  % Robust estimate of noise standard deviation
    threshold = sigma * sqrt(2*log(length(noisySignal))); % VisuShrink threshold

    C_denoised = wthresh(C, threshold_type, threshold);  % Apply thresholding.  Now uses 's' or 'h'

    enhancedSignal = waverec(C_denoised, L, wavelet_family);  % Reconstruct the denoised signal

    % Ensure output is the same length as input.
    enhancedSignal = enhancedSignal(1:length(noisySignal));
end