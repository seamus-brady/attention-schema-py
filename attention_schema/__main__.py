from .controller import Controller


def main() -> None:
    controller = Controller()
    controller.loop()


if __name__ == "__main__":
    main()
