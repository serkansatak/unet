from src import Config, ModelTrainer, BaseDataset, UNet, Tester
import prettyprinter as pp
import argparse


def main():
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        optional=True,
        help="Path to single test file. Test mode configured in config.py",
    )
    args = parser.parse_args()

    pp.pprint(args)
    pp.pprint(config.to_dict())

    if config.general.mode == "train":
        model_trainer = ModelTrainer(config)
        model_trainer.train()
    elif config.general.mode == "test":
        tester = Tester(config)
        if args.test_file:
            tester.test_single(args.test_file)
        else:
            tester.test_batch()


if __name__ == "__main__":
    main()
