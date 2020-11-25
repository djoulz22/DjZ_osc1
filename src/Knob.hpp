#include "componentlibrary.hpp"

// using namespace rack;
using namespace componentlibrary;

// namespace djoulz {
    // namespace knob {
        
        struct RoundBigKnob : RoundKnob {
            RoundBigKnob() {
                setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/RoundBigKnob.svg")));
            }
        };

        struct RoundSmallKnob : RoundKnob {
            RoundSmallKnob() {
                setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/RoundSmallKnob.svg")));
            }
        };

        struct RoundTestKnob : RoundKnob {
            RoundTestKnob() {
                setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/RoundTestKnob.svg")));
            }
        };

        struct RoundDKnob : RoundKnob {
            RoundDKnob() {
                setSvg(APP->window->loadSvg(asset::plugin(pluginInstance, "res/RoundDKnob.svg")));
            }
        };

        struct RoundSnapDKnob : RoundDKnob {
            RoundSnapDKnob() {
                snap = true;
            }
        };
    // }
// }