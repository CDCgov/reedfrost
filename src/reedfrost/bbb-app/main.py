from app import Controller

root = "src/reedfrost/bbb-app/"


def main(
    data_yaml=f"{root}resources/data.yaml",
    ui_yaml=f"{root}resources/ui.yaml",
    text_yaml=f"{root}resources/text.yaml",
):
    Controller(data_yaml, ui_yaml, text_yaml)


if __name__ == "__main__":
    main()
