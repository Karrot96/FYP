"""[summary]
"""
import argparse
import logging as log
import time
import sys
import numpy as np
import cv2
import video
import matplotlib.pyplot as plt
import winsound
np.set_printoptions(threshold=sys.maxsize)


def from_video(cap, out=None):
    """[summary]

    Arguments:
        cap {[type]} -- [description]

    Keyword Arguments:
        out {[type]} -- [description] (default: {None})
    """
    times = []
    frame = 0
    log.debug("resolution: %s", (int(cap.get(3)), int(cap.get(4))))
    if out:
        out = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
            10,
            (int(cap.get(3)), int(cap.get(4)))
            )
    vid = video.Video(cap, out)
    ret = 1
    while(vid.cap.isOpened and ret == 1):
        start = time.perf_counter()
        log.info("frame: %s", frame)
        ret = vid.shoelace_finding(frame)
        if ret == 0:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame += 1
        end = time.perf_counter()
        log.info("Current Total time is %s resulting in frame rate of %s",
                 end-start, 1/(end-start)
                 )
        times.append(end-start)
    fig = plt.figure()
    plt.hist(times, bins=25,normed=True)
    plt.title("Test 1 time per a frame histogram")
    plt.ylabel("Probability(%)")
    plt.xlabel("Time per frame(s)")
    fig.savefig('Test1.png')
    cap.release()
    cv2.destroyAllWindows()
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 250  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    exit()


def from_camera(cap, out=None):
    """Take input from webcam
    Arguments:
        cap {cv2.VideoCapture} -- The input videon  stream in this case
                                  a webcam
    Keyword Arguments:
        out {string} -- An output video file location (default: {None})
    """
    log.debug("resolution: %s", (int(cap.get(3)), int(cap.get(4))))
    if out:
        out = cv2.VideoWriter(
            out,
            cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),
            10,
            (int(cap.get(3)), int(cap.get(4)))
            )

    vid = video.Video(cap, out)
    frame = 0
    while True:
        vid.shoelace_finding(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame += 1
    cap.release()
    cv2.destroyAllWindows()


def video_input():
    """ Used to handle argument parsing, picks the correct input source
        or file as well as debug mode
    """
    parser = argparse.ArgumentParser(description='Detect and track shoelaces')
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        help="Store input camera number if different from default"
        )
    parser.add_argument(
        "-v",
        "--video",
        action="store",
        help="location of the input video file"
        )
    parser.add_argument(
        "-d",
        "--debug",
        action="store",
        help="Used to enter debug mode. Using argument \
                'console' will output to console, any \
                other argument will be a file path for \
                the debug and log information"
        )
    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Turns on log. Debug mode wont work without\
                this being enabled"
        )
    parser.add_argument(
        "-r",
        "--record",
        action="store",
        help="Records output video to filename. Only \
                works in log mode"
        )
    args = parser.parse_args()
    if args.log:
        if args.debug == 'console':
            log.basicConfig(
                level=log.DEBUG,
                format='%(asctime)s, %(levelname)s, %(message)s'
                )
            log.debug("Debug turned on. Outputting to console")
        elif args.debug:
            log.basicConfig(
                level=log.DEBUG,
                filename=args.debug,
                filemode='w',
                format='%(asctime)s, %(levelname)s , %(message)s'
                )
            log.debug("Debug turned on. Outputting to %s", args.debug)
        else:
            log.basicConfig(
                level=log.INFO,
                format='%(asctime)s, %(levelname)s , %(message)s'
                )
            log.info("log started")
        if args.record:
            if args.input:
                log.info("Selecting Camera %s", args.input)
                from_camera(cv2.VideoCapture(args.input), args.record)
            elif args.video:
                log.info("Using %s as video", args.video)
                from_video(cv2.VideoCapture(args.video), args.record)
            else:
                log.info("Selecting Camera 0")
                from_camera(cv2.VideoCapture(0), args.record)
    if args.input:
        log.info("Selecting Camera %s", args.input)
        from_camera(cv2.VideoCapture(args.input))
    elif args.video:
        log.info("Using %s as video", args.video)
        from_video(cv2.VideoCapture(args.video))
    else:
        log.warning("No webcam Selected. Selecting Camera 0")
        from_camera(cv2.VideoCapture(0))


def main():
    """[summary]
    """
    video_input()

if __name__ == "__main__":
    main()
