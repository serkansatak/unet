from src import Config, ModelTrainer, BaseDataset, UNet, Tester
import prettyprinter as pp


def main():
    config = Config()

    pp.pprint(config.to_dict())

    if config.general.mode == "train":
        model_trainer = ModelTrainer(config)
        model_trainer.train()
    elif config.general.mode == "test":
        tester = Tester(config)
        if config.general.single_test_file:
            tester.test_single(config.general.single_test_file)
        else:
            tester.test_batch()


if __name__ == "__main__":
    main()
