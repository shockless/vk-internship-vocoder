from src.stretch import Stretch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='phase_vocoder')
    parser.add_argument('--input', default='.\\data\\test_mono.wav', type=str, help='<input.wav> path')
    parser.add_argument('--output', default='.\\data\\test_mono_output.wav', type=str, help='<output.wav> path')
    parser.add_argument('--time_stretch_ratio', default=1, type=float, help='<time_stretch_ratio>')
    arguments = parser.parse_args()
    stretcher = Stretch(arguments.input)
    stretcher.stretch(arguments.time_stretch_ratio)
    stretcher.write(arguments.output)