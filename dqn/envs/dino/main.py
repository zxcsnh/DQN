__author__ = "Shivam Shekhar"

from env import introscreen, play_human


def main():
    isGameQuit = introscreen()
    if not isGameQuit:
        play_human()


if __name__ == "__main__":
    main()
