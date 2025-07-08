//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AudioCapture.hpp"
#include "tensorflow/lite/micro/micro_log.h"

namespace audio
{
    std::vector<float> AudioCapture::LoadAudioFile(std::string filePath)
    {
        // NOT supported
        MicroPrintf("Error, not supported.");
        return {};
    }

    void AudioCapture::InitSlidingWindow(float* data, size_t dataSize, int minSamples, size_t stride)
    {
        this->m_window = SlidingWindow<const float>(data, dataSize, minSamples, stride);
    }

    bool AudioCapture::HasNext()
    {
        return m_window.HasNext();
    }

    std::vector<float> AudioCapture::Next()
    {
        if (this->m_window.HasNext())
        {
            int remainingData = this->m_window.RemainingData();
            const float* windowData = this->m_window.Next();

            size_t windowSize = this->m_window.GetWindowSize();

            if(remainingData < windowSize)
            {
                std::vector<float> audioData(windowSize, 0.0f);
                for(int i = 0; i < remainingData; ++i)
                {
                    audioData[i] = *windowData;
                    if(i < remainingData - 1)
                    {
                        ++windowData;
                    }
                }
                return audioData;
            }
            else
            {
                std::vector<float> audioData(windowData, windowData + windowSize);
                return audioData;
            }
        }
        else
        {
            MicroPrintf("Error, end of audio data reached.");
            return {};
        }
    }
} //namespace asr