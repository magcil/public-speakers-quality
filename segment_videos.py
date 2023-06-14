import contextlib
import argparse
import random
import wave
import os

def segment(input, output):
    with contextlib.closing(wave.open(input, 'r')) as wa:
        frames = wa.getnframes()
        rate = wa.getframerate()
        duration = int(frames/float(rate))
    if (duration > 90):
        s1 = int(duration/3)
        s2 = int((duration/3)*2)
        f = random.randint(0, s1-30)
        s = random.randint(s1, s2-30)
        t = random.randint(s2, duration-30)

        if os.path.exists(output):
            return
        else:
            first = "ffmpeg -ss " + str(f) + " -to " + str(f+30) + " -i " + input + " " + "./first.wav"
            second = "ffmpeg -ss " + str(s) + " -to " + str(s+30) + " -i " + input + " " + "./second.wav"
            third = "ffmpeg -ss " + str(t) + " -to " + str(t+30) + " -i " + input + " " + "./third.wav"
            final = "ffmpeg -i first.wav -i second.wav -i third.wav -filter_complex '[0:0] [1:0]concat=n=3:v=0:a=1[out]' -map '[out]' " + output
            #concatenate clips
            os.system(first)
            os.system(second)
            os.system(third)
            os.system(final)
            os.system("rm first.wav second.wav third.wav")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', required=True)
    parser.add_argument('-n', '--n_segments', required=True)
    parser.add_argument('-d', '--segment_duration', required=True)
    parser.add_argument('-o', '--output', required=False)

    args = parser.parse_args()
    if (args.n_segments != "3") or (args.segment_duration != "30"):
        raise BaseException("Wrong parameters")

    input = args.input

    if os.path.isdir(input):			#input is folder
        for video in os.scandir(input):		#iterate videos in folders
            #open wav file
            if (input[0] == '.'):
                path = os.path.join('', video)
            else:
                path = os.path.join(input, video)
            output = input + '/new_' + str(video).split("'")[1]
            segment(path, output)
    else:
        if (args.output == None):
            raise BaseException("Output file is required")
        output = args.output
        if (output.split('.').pop() != "wav"):
            output = output + '.wav'
        segment(input, output)
