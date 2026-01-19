import argparse


def main():
    parser = argparse.ArgumentParser(description='Print config values for shell scripts.')
    parser.add_argument('--print_task', action='store_true', help='print task data_dir name')
    args = parser.parse_args()

    if args.print_task:
        from .config import Config
        cfg = Config()
        print(cfg.dataset_settings[cfg.task]['data_dir'])


if __name__ == '__main__':
    main()
