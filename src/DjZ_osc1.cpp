#include "plugin.hpp"
#include "dsp/oscillator.hpp"
#include "dsp/pitch.hpp"

using namespace djoulz::dsp;

using simd::float_4;


template <typename T>
struct LowFrequencyOscillator {
	T phase = 0.f;
	T pw = 0.5f;
	T freq = 1.f;
	bool invert = false;
	bool bipolar = false;
	T resetState = T::mask();

	void setPitch(T pitch) {
		pitch = simd::fmin(pitch, 10.f);
		freq = dsp::approxExp2_taylor5(pitch + 30) / 1073741824;
	}
	void setPulseWidth(T pw) {
		const T pwMin = 0.01f;
		this->pw = clamp(pw, pwMin, 1.f - pwMin);
	}
	void setReset(T reset) {
		reset = simd::rescale(reset, 0.1f, 2.f, 0.f, 1.f);
		T on = (reset >= 1.f);
		T off = (reset <= 0.f);
		T triggered = ~resetState & on;
		resetState = simd::ifelse(off, 0.f, resetState);
		resetState = simd::ifelse(on, T::mask(), resetState);
		phase = simd::ifelse(triggered, 0.f, phase);
	}
	void step(float dt) {
		T deltaPhase = simd::fmin(freq * dt, 0.5f);
		phase += deltaPhase;
		phase -= (phase >= 1.f) & 1.f;
	}
	T sin() {
		T p = phase;
		if (!bipolar)
			p -= 0.25f;
		T v = simd::sin(2 * M_PI * p);
		if (invert)
			v *= -1.f;
		if (!bipolar)
			v += 1.f;
		return v;
	}
	T tri() {
		T p = phase;
		if (bipolar)
			p += 0.25f;
		T v = 4.f * simd::fabs(p - simd::round(p)) - 1.f;
		if (invert)
			v *= -1.f;
		if (!bipolar)
			v += 1.f;
		return v;
	}
	T saw() {
		T p = phase;
		if (!bipolar)
			p -= 0.5f;
		T v = 2.f * (p - simd::round(p));
		if (invert)
			v *= -1.f;
		if (!bipolar)
			v += 1.f;
		return v;
	}
	T sqr() {
		T v = simd::ifelse(phase < pw, 1.f, -1.f);
		if (invert)
			v *= -1.f;
		if (!bipolar)
			v += 1.f;
		return v;
	}
	T light() {
		return simd::sin(2 * T(M_PI) * phase);
	}
};

// Accurate only on [0, 1]
template <typename T>
T sin2pi_pade_05_7_6(T x) {
	x -= 0.5f;
	return (T(-6.28319) * x + T(35.353) * simd::pow(x, 3) - T(44.9043) * simd::pow(x, 5) + T(16.0951) * simd::pow(x, 7))
	       / (1 + T(0.953136) * simd::pow(x, 2) + T(0.430238) * simd::pow(x, 4) + T(0.0981408) * simd::pow(x, 6));
}

template <typename T>
T sin2pi_pade_05_5_4(T x) {
	x -= 0.5f;
	return (T(-6.283185307) * x + T(33.19863968) * simd::pow(x, 3) - T(32.44191367) * simd::pow(x, 5))
	       / (1 + T(1.296008659) * simd::pow(x, 2) + T(0.7028072946) * simd::pow(x, 4));
}

template <typename T>
T expCurve(T x) {
	return (3 + x * (-13 + 5 * x)) / (3 + 2 * x);
}

template <int OVERSAMPLE, int QUALITY, typename T>
struct VoltageControlledOscillator {
	bool analog = false;
	bool soft = false;
	bool syncEnabled = false;
	// For optimizing in serial code
	int channels = 0;

	T lastSyncValue = 0.f;
	T phase = 0.f;
	T freq;
	T pulseWidth = 0.5f;
	T syncDirection = 1.f;

	dsp::TRCFilter<T> sqrFilter;

	dsp::MinBlepGenerator<QUALITY, OVERSAMPLE, T> sqrMinBlep;
	dsp::MinBlepGenerator<QUALITY, OVERSAMPLE, T> sawMinBlep;
	dsp::MinBlepGenerator<QUALITY, OVERSAMPLE, T> triMinBlep;
	dsp::MinBlepGenerator<QUALITY, OVERSAMPLE, T> sinMinBlep;

	T sqrValue = 0.f;
	T sawValue = 0.f;
	T triValue = 0.f;
	T sinValue = 0.f;

	void setPitch(T pitch) {
		freq = dsp::FREQ_C4 * dsp::approxExp2_taylor5(pitch + 30) / 1073741824;
	}

	void setPulseWidth(T pulseWidth) {
		const float pwMin = 0.01f;
		this->pulseWidth = simd::clamp(pulseWidth, pwMin, 1.f - pwMin);
	}

	void process(float deltaTime, T syncValue) {
		// Advance phase
		T deltaPhase = simd::clamp(freq * deltaTime, 1e-6f, 0.35f);
		if (soft) {
			// Reverse direction
			deltaPhase *= syncDirection;
		}
		else {
			// Reset back to forward
			syncDirection = 1.f;
		}
		phase += deltaPhase;
		// Wrap phase
		phase -= simd::floor(phase);

		// Jump sqr when crossing 0, or 1 if backwards
		T wrapPhase = (syncDirection == -1.f) & 1.f;
		T wrapCrossing = (wrapPhase - (phase - deltaPhase)) / deltaPhase;
		int wrapMask = simd::movemask((0 < wrapCrossing) & (wrapCrossing <= 1.f));
		if (wrapMask) {
			for (int i = 0; i < channels; i++) {
				if (wrapMask & (1 << i)) {
					T mask = simd::movemaskInverse<T>(1 << i);
					float p = wrapCrossing[i] - 1.f;
					T x = mask & (2.f * syncDirection);
					sqrMinBlep.insertDiscontinuity(p, x);
				}
			}
		}

		// Jump sqr when crossing `pulseWidth`
		T pulseCrossing = (pulseWidth - (phase - deltaPhase)) / deltaPhase;
		int pulseMask = simd::movemask((0 < pulseCrossing) & (pulseCrossing <= 1.f));
		if (pulseMask) {
			for (int i = 0; i < channels; i++) {
				if (pulseMask & (1 << i)) {
					T mask = simd::movemaskInverse<T>(1 << i);
					float p = pulseCrossing[i] - 1.f;
					T x = mask & (-2.f * syncDirection);
					sqrMinBlep.insertDiscontinuity(p, x);
				}
			}
		}

		// Jump saw when crossing 0.5
		T halfCrossing = (0.5f - (phase - deltaPhase)) / deltaPhase;
		int halfMask = simd::movemask((0 < halfCrossing) & (halfCrossing <= 1.f));
		if (halfMask) {
			for (int i = 0; i < channels; i++) {
				if (halfMask & (1 << i)) {
					T mask = simd::movemaskInverse<T>(1 << i);
					float p = halfCrossing[i] - 1.f;
					T x = mask & (-2.f * syncDirection);
					sawMinBlep.insertDiscontinuity(p, x);
				}
			}
		}

		// Detect sync
		// Might be NAN or outside of [0, 1) range
		if (syncEnabled) {
			T deltaSync = syncValue - lastSyncValue;
			T syncCrossing = -lastSyncValue / deltaSync;
			lastSyncValue = syncValue;
			T sync = (0.f < syncCrossing) & (syncCrossing <= 1.f) & (syncValue >= 0.f);
			int syncMask = simd::movemask(sync);
			if (syncMask) {
				if (soft) {
					syncDirection = simd::ifelse(sync, -syncDirection, syncDirection);
				}
				else {
					T newPhase = simd::ifelse(sync, (1.f - syncCrossing) * deltaPhase, phase);
					// Insert minBLEP for sync
					for (int i = 0; i < channels; i++) {
						if (syncMask & (1 << i)) {
							T mask = simd::movemaskInverse<T>(1 << i);
							float p = syncCrossing[i] - 1.f;
							T x;
							x = mask & (sqr(newPhase) - sqr(phase));
							sqrMinBlep.insertDiscontinuity(p, x);
							x = mask & (saw(newPhase) - saw(phase));
							sawMinBlep.insertDiscontinuity(p, x);
							x = mask & (tri(newPhase) - tri(phase));
							triMinBlep.insertDiscontinuity(p, x);
							x = mask & (sin(newPhase) - sin(phase));
							sinMinBlep.insertDiscontinuity(p, x);
						}
					}
					phase = newPhase;
				}
			}
		}

		// Square
		sqrValue = sqr(phase);
		sqrValue += sqrMinBlep.process();

		if (analog) {
			sqrFilter.setCutoffFreq(20.f * deltaTime);
			sqrFilter.process(sqrValue);
			sqrValue = sqrFilter.highpass() * 0.95f;
		}

		// Saw
		sawValue = saw(phase);
		sawValue += sawMinBlep.process();

		// Tri
		triValue = tri(phase);
		triValue += triMinBlep.process();

		// Sin
		sinValue = sin(phase);
		sinValue += sinMinBlep.process();
	}

	T sin(T phase) {
		T v;
		if (analog) {
			// Quadratic approximation of sine, slightly richer harmonics
			T halfPhase = (phase < 0.5f);
			T x = phase - simd::ifelse(halfPhase, 0.25f, 0.75f);
			v = 1.f - 16.f * simd::pow(x, 2);
			v *= simd::ifelse(halfPhase, 1.f, -1.f);
		}
		else {
			v = sin2pi_pade_05_5_4(phase);
			// v = sin2pi_pade_05_7_6(phase);
			// v = simd::sin(2 * T(M_PI) * phase);
		}
		return v;
	}
	T sin() {
		return sinValue;
	}

	T tri(T phase) {
		T v;
		if (analog) {
			T x = phase + 0.25f;
			x -= simd::trunc(x);
			T halfX = (x >= 0.5f);
			x *= 2;
			x -= simd::trunc(x);
			v = expCurve(x) * simd::ifelse(halfX, 1.f, -1.f);
		}
		else {
			v = 1 - 4 * simd::fmin(simd::fabs(phase - 0.25f), simd::fabs(phase - 1.25f));
		}
		return v;
	}
	T tri() {
		return triValue;
	}

	T saw(T phase) {
		T v;
		T x = phase + 0.5f;
		x -= simd::trunc(x);
		if (analog) {
			v = -expCurve(x);
		}
		else {
			v = 2 * x - 1;
		}
		return v;
	}

	T saw() {
		return sawValue;
	}

	T sqr(T phase) {
		T v = simd::ifelse(phase < pulseWidth, 1.f, -1.f);
		return v;
	}
	T sqr() {
		return sqrValue;
	}

	T light() {
		return simd::sin(2 * T(M_PI) * phase);
	}
};

template <typename EnumT, typename BaseEnumT>
class InheritEnum
{
public:
  InheritEnum() {}
  InheritEnum(EnumT e)
    : enum_(e)
  {}

  InheritEnum(BaseEnumT e)
    : baseEnum_(e)
  {}

  explicit InheritEnum( int val )
    : enum_(static_cast<EnumT>(val))
  {}

  operator EnumT() const { return enum_; }
private:
  // Note - the value is declared as a union mainly for as a debugging aid. If 
  // the union is undesired and you have other methods of debugging, change it
  // to either of EnumT and do a cast for the constructor that accepts BaseEnumT.
  union
  { 
    EnumT enum_;
    BaseEnumT baseEnum_;
  };
};

struct DjZ_osc1 : Module {
	enum ParamIds {
		FREQ_PARAM,
		PITCH_PARAM,
		GLOBAL_FINE_PARAM,
		// OSC1
		OSC1_OCTAVE_PARAM,
		OSC1_WAVE_PARAM,
		// OSC2
		OSC2_OCTAVE_PARAM,
		OSC2_FINE_PARAM,
		OSC2_WAVE_PARAM,
		// OSC3
		OSC3_OCTAVE_PARAM,
		OSC3_FINE_PARAM,
		OSC3_WAVE_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		PITCH_INPUT,
		OSC1_WAVE_INPUT,
		OSC2_WAVE_INPUT,
		OSC3_WAVE_INPUT,
		OSC1_OCTAVE_INPUT,
		OSC2_OCTAVE_INPUT,
		OSC3_OCTAVE_INPUT,
		SYNC_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		OUT_OSC1_OUTPUT,
		OUT_OSC2_OUTPUT,
		OUT_OSC3_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		NUM_LIGHTS
	};

	enum WaveOsc1_2 {
		TRIANGLE_WAVE,
		TRIANGLESAW_WAVE,
		SAW_WAVE,
		SQUARE_WAVE,
		PULSE_25_WAVE,
		PULSE_10_WAVE
	};

	enum WaveOsc3 {
		TRIANGLE_WAVE2,
		REVERSESAW_WAVE,
		SAW_WAVE2,
		SQUARE_WAVE2,
		PULSE_25_WAVE2,
		PULSE_10_WAVE2
	};

	// enum WaveBase {
	// 	TRIANGLE_WAVE
	// };

	// enum LastWaveBase {
	// 	SAW_WAVE,
	// 	SQUARE_WAVE,
	// 	PULSE_25_WAVE,
	// 	PULSE_10_WAVE
	// };

	// enum SecondWave_osc1_2 {		
	// 	TRIANGLESAW_WAVE	
	// };

	// enum SecondWave_osc3 {		
	// 	REVERSESAW_WAVE	
	// };

	// typedef InheritEnum< WaveBase,SecondWave_osc1_2 > WaveOsc1_2_start;
	// typedef InheritEnum< LastWaveBase,WaveOsc1_2_start > WaveOsc1_2;
	// typedef InheritEnum< WaveBase,SecondWave_osc3 > WaveOsc3_start;
	// typedef InheritEnum< LastWaveBase,WaveOsc3_start > WaveOsc3;

	enum OctaveOsc1_2 {
		OCTLO,
		OCT32,
		OCT16,
		OCT8,
		OCT4,
		OCT2
	};

	enum SpecOctave {
		OCTLOMONO
	};

	typedef InheritEnum< SpecOctave, OctaveOsc1_2 > OctaveOsc3;	

	float _phaseOsc1 = 0.f;
	float _phaseOsc2 = 0.f;
	float _phaseOsc3 = 0.f;
	float _fine = 0.f;
	float _fineOsc2 = 0.f;
	float _fineOsc3 = 0.f;

	// WaveOsc1_2 _waveOSC1 = TRIANGLE_WAVE;
	// WaveOsc1_2 _waveOSC2 = TRIANGLE_WAVE;
	// WaveOsc3 _waveOSC3 = TRIANGLE_WAVE;

	// float _octaveOSC1 = OCT8*2.0f;
	// float _octaveOSC2 = OCT4*2.0f;
	// float _octaveOSC3 = OCT16*2.0f;

	float _waveOSC1 = 0.f;
	float _waveOSC2 = 0.f;
	float _waveOSC3 = 0.f;

	float _octaveOSC1 = 0.f;
	float _octaveOSC2 = 0.f;
	float _octaveOSC3 = 0.f;

	float _freqParamOsc1 = rack::dsp::FREQ_C4;
	float _freqParamOsc2 = rack::dsp::FREQ_C4;
	float _freqParamOsc3 = rack::dsp::FREQ_C4;

	// VoltageControlledOscillator<16, 16, float_4> oscillators[4];
	// dsp::ClockDivider lightDivider;

	DjZ_osc1() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(PITCH_PARAM, -4.f, 4.f, 0.f, "");

		configParam(FREQ_PARAM, -54.f, 54.f, _freqParamOsc1, "Frequency", " Hz", dsp::FREQ_SEMITONE, dsp::FREQ_C4);

		configParam(GLOBAL_FINE_PARAM, -1.f, 1.f, _fine, "Global Fine frequency");
		configParam(OSC2_FINE_PARAM, -1.f, 1.f, _fineOsc2, "Fine frequency Osc2");
		configParam(OSC3_FINE_PARAM, -1.f, 1.f, _fineOsc3, "Fine frequency Osc3");		

		configParam(OSC1_OCTAVE_PARAM, -4.f, 4.f, _octaveOSC1, "Octave Osc1");
		configParam(OSC2_OCTAVE_PARAM, -4.f, 4.f, _octaveOSC2, "Octave Osc2");
		configParam(OSC3_OCTAVE_PARAM, -4.f, 4.f, _octaveOSC3, "Octave Osc3");

		configParam(OSC1_WAVE_PARAM, 0, 5, _waveOSC1, "Waveform Osc1");
		configParam(OSC2_WAVE_PARAM, 0, 5, _waveOSC2, "Waveform Osc2");
		configParam(OSC3_WAVE_PARAM, 0, 5, _waveOSC3, "Waveform Osc3");
	}

	bool active() {
		return outputs[OUT_OSC1_OUTPUT].isConnected() || outputs[OUT_OSC2_OUTPUT].isConnected() || outputs[OUT_OSC3_OUTPUT].isConnected();
	}

	void processAlways(const ProcessArgs& args) {
		// _waveOSC1 = (WaveOsc1_2)inputs[OSC1_WAVE_INPUT].getVoltage();
		// _waveOSC2 = (WaveOsc1_2)inputs[OSC2_WAVE_INPUT].getVoltage();
		// _waveOSC3 = (WaveOsc3)inputs[OSC3_WAVE_INPUT].getVoltage();

		_waveOSC1 = params[OSC1_WAVE_PARAM].getValue();
		_waveOSC2 = params[OSC2_WAVE_PARAM].getValue();
		_waveOSC3 = params[OSC3_WAVE_PARAM].getValue();

		_octaveOSC1 = params[OSC1_OCTAVE_PARAM].getValue();
		_octaveOSC2 = params[OSC2_OCTAVE_PARAM].getValue();
		_octaveOSC3 = params[OSC3_OCTAVE_PARAM].getValue();
	}

	VoltageControlledOscillator<16, 16, float_4> oscillators_1[4];
	VoltageControlledOscillator<16, 16, float_4> oscillators_2[4];
	VoltageControlledOscillator<16, 16, float_4> oscillators_3[4];

	void process(const ProcessArgs& args) override {

		processAlways(args);

		if (active()) {
			// const float referencePitch = 0.f;
			// const float referenceOctave = OCT16*2.0f;

			float freqParam = params[FREQ_PARAM].getValue() / 12.f;
			freqParam += rack::dsp::quadraticBipolar(params[GLOBAL_FINE_PARAM].getValue()) * 3.f / 12.f;

			// float pitch = params[PITCH_PARAM].getValue();
			// pitch += inputs[PITCH_INPUT].getVoltage();
			// pitch = clamp(pitch, -4.f, 4.f);

			_freqParamOsc1 = 0.f; //* std::pow(2.f, pitch);
			_freqParamOsc2 = 0.f; //* std::pow(2.f, pitch);
			_freqParamOsc3 = 0.f; //* std::pow(2.f, pitch);

			_freqParamOsc2 += rack::dsp::quadraticBipolar(params[OSC2_FINE_PARAM].getValue()) * 3.f / 12.f;
			_freqParamOsc3 += rack::dsp::quadraticBipolar(params[OSC3_FINE_PARAM].getValue()) * 3.f / 12.f;

			int channels = std::max(inputs[PITCH_INPUT].getChannels(), 1);
			
			// The default pitch is C4 = 261.6256f
			// float freq = dsp::FREQ_C4 * std::pow(2.f, pitch);

			// for (int osc = 0; osc < 3; osc++){	
			for (int i=0; i<3; i++) {
				for (int c = 0; c < channels; c += 4) {							
					float pitch = inputs[PITCH_INPUT].getVoltage(c);
					
					if (i == 0 && outputs[OUT_OSC1_OUTPUT].isConnected()) {
						auto* oscillator = &oscillators_1[c / 4];
						oscillator->channels = std::min(channels - c, 4);
						oscillator->analog = true;
						oscillator->soft = false;
						oscillator->syncEnabled = false;

						float waveOSC1 = roundf(_waveOSC1 + inputs[OSC1_WAVE_INPUT].getPolyVoltage(c));

						if (waveOSC1 == (float)SQUARE_WAVE) {
							oscillator->setPulseWidth(0.5f);
						}
						else if (waveOSC1 == (float)PULSE_25_WAVE) {
							oscillator->setPulseWidth(0.25f);
						}
						else if (waveOSC1 == (float)PULSE_10_WAVE) {
							oscillator->setPulseWidth(0.1f);
						}
						else {
							oscillator->setPulseWidth(0.f);
						}

						float octave_osc1 = _freqParamOsc1 + roundf(_octaveOSC1 + inputs[OSC1_OCTAVE_INPUT].getPolyVoltage(c));
						float pitchOsc1 = pitch + octave_osc1;						
						oscillator->setPitch(pitchOsc1);									
						oscillator->process(args.sampleTime, 0.f);

						if (waveOSC1 == (float)TRIANGLE_WAVE) {
							outputs[OUT_OSC1_OUTPUT].setVoltageSimd(oscillator->tri(), c);	
						}
						else if (waveOSC1 == (float)TRIANGLESAW_WAVE) {
							outputs[OUT_OSC1_OUTPUT].setVoltageSimd(oscillator->tri()+oscillator->saw(), c);	
							// outputs[OUT_OSC1_OUTPUT].setVoltageSimd(oscillator->saw(), c);	
						}
						else if (waveOSC1 == (float)SAW_WAVE) {
							outputs[OUT_OSC1_OUTPUT].setVoltageSimd(oscillator->saw(), c);
						}
						else if ( (waveOSC1 == (float)SQUARE_WAVE) || (waveOSC1 == (float)PULSE_25_WAVE) || (waveOSC1 == (float)PULSE_10_WAVE))  {
							outputs[OUT_OSC1_OUTPUT].setVoltageSimd(oscillator->sqr(), c);
						}
						else {
							
						}

						outputs[OUT_OSC1_OUTPUT].setChannels(channels);		
					}

					if (i == 1 && outputs[OUT_OSC2_OUTPUT].isConnected()) {
						auto* oscillator = &oscillators_2[c / 4];
						oscillator->channels = std::min(channels - c, 4);
						oscillator->analog = true;
						oscillator->soft = false;
						oscillator->syncEnabled = false;
						
						float waveOSC2 = roundf(_waveOSC2 + inputs[OSC2_WAVE_INPUT].getPolyVoltage(c));

						if (waveOSC2 == (float)SQUARE_WAVE) {
							oscillator->setPulseWidth(0.5f);
						}
						else if (waveOSC2 == (float)PULSE_25_WAVE) {
							oscillator->setPulseWidth(0.25f);
						}
						else if (waveOSC2 == (float)PULSE_10_WAVE) {
							oscillator->setPulseWidth(0.1f);
						}
						else {
							oscillator->setPulseWidth(0.f);
						}

						float octave_osc2 = _freqParamOsc2 + roundf(_octaveOSC2 + inputs[OSC2_OCTAVE_INPUT].getPolyVoltage(c));
						float pitchOsc2 = pitch + octave_osc2;
						oscillator->setPitch(pitchOsc2);									
						oscillator->process(args.sampleTime, 0.f);
						
						if (waveOSC2 == (float)TRIANGLE_WAVE) {
							outputs[OUT_OSC2_OUTPUT].setVoltageSimd(oscillator->tri(), c);	
						}
						else if (waveOSC2 == (float)TRIANGLESAW_WAVE) {
							outputs[OUT_OSC2_OUTPUT].setVoltageSimd(oscillator->tri()+oscillator->saw(), c);	
							// outputs[OUT_OSC1_OUTPUT].setVoltageSimd(oscillator->saw(), c);	
						}
						else if (waveOSC2 == (float)SAW_WAVE) {
							outputs[OUT_OSC2_OUTPUT].setVoltageSimd(oscillator->saw(), c);
						}
						else if ( (waveOSC2 == (float)SQUARE_WAVE) || (waveOSC2 == (float)PULSE_25_WAVE) || (waveOSC2 == (float)PULSE_10_WAVE))  {
							outputs[OUT_OSC2_OUTPUT].setVoltageSimd(oscillator->sqr(), c);
						}
						else {
							
						}

						outputs[OUT_OSC2_OUTPUT].setChannels(channels);

					}
					
					if (i == 2 && outputs[OUT_OSC3_OUTPUT].isConnected()) {
						auto* oscillator = &oscillators_3[c / 4];
						oscillator->channels = std::min(channels - c, 4);
						oscillator->analog = true;
						oscillator->soft = false;
						oscillator->syncEnabled = false;
						
						float waveOSC3 = roundf(_waveOSC3 + inputs[OSC3_WAVE_INPUT].getPolyVoltage(c));

						if (waveOSC3 == (float)SQUARE_WAVE2) {
							oscillator->setPulseWidth(0.5f);
						}
						else if (waveOSC3 == (float)PULSE_25_WAVE2) {
							oscillator->setPulseWidth(0.25f);
						}
						else if (waveOSC3 == (float)PULSE_10_WAVE2) {
							oscillator->setPulseWidth(0.1f);
						}
						else {
							oscillator->setPulseWidth(0.f);
						}

						float octave_osc3 = _freqParamOsc3 + roundf(_octaveOSC3 + inputs[OSC3_OCTAVE_INPUT].getPolyVoltage(c));
						float pitchOsc3 = pitch + octave_osc3;
						oscillator->setPitch(pitchOsc3);									
						oscillator->process(args.sampleTime, 0.f);

						if (waveOSC3 == (float)TRIANGLE_WAVE2) {
							outputs[OUT_OSC3_OUTPUT].setVoltageSimd(oscillator->tri(), c);	
						}
						else if (waveOSC3 == (float)SAW_WAVE2) {
							outputs[OUT_OSC3_OUTPUT].setVoltageSimd(oscillator->saw(), c);
						}
						else if (waveOSC3 == (float)REVERSESAW_WAVE) {
							outputs[OUT_OSC3_OUTPUT].setVoltageSimd(-oscillator->saw(), c);
						}
						else if ( (waveOSC3 == (float)SQUARE_WAVE2) || (waveOSC3 == (float)PULSE_25_WAVE2) || (waveOSC3 == (float)PULSE_10_WAVE2))  {
							outputs[OUT_OSC3_OUTPUT].setVoltageSimd(oscillator->sqr(), c);
						}
						else {
							
						}

						outputs[OUT_OSC3_OUTPUT].setChannels(channels);
					}									
				}
			// } 
			}				
		}
	}
};


struct DjZ_osc1Widget : ModuleWidget {
	DjZ_osc1Widget(DjZ_osc1* module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/DjZ_osc1.svg")));

		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		// addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(24.1, 25.26)), module, DjZ_osc1::PITCH_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(24.1, 25.26)), module, DjZ_osc1::PITCH_INPUT));

		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(25, 10.578)), module, DjZ_osc1::OSC1_OCTAVE_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(25, 50.578)), module, DjZ_osc1::OSC2_OCTAVE_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(25, 100.578)), module, DjZ_osc1::OSC3_OCTAVE_PARAM));

		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(70, 50.578)), module, DjZ_osc1::OSC2_FINE_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(70, 100.578)), module, DjZ_osc1::OSC3_FINE_PARAM));

		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(100, 10.578)), module, DjZ_osc1::OSC1_WAVE_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(100, 50.578)), module, DjZ_osc1::OSC2_WAVE_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(100, 100.578)), module, DjZ_osc1::OSC3_WAVE_PARAM));

		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(120, 10.578)), module, DjZ_osc1::OUT_OSC1_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(120, 50.578)), module, DjZ_osc1::OUT_OSC2_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(120, 100.578)), module, DjZ_osc1::OUT_OSC3_OUTPUT));
	}
};


Model* modelDjZ_osc1 = createModel<DjZ_osc1, DjZ_osc1Widget>("DjZ_osc1");