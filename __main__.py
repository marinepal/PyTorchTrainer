import argparse

from src.pytorch_trainer.pytorchTrainer import CifarPytorchTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training on CIFAR dataset.")
    parser.add_argument('--model_name', '-mn', help='model you want to use', required=True)
    parser.add_argument('--metric', '-m', help="output one metric", default='f1')
    parser.add_argument('--output_json_path', '-o', help='output json path', default='results.json')
    parser.add_argument('--lr', help='learning rate', default=0.1)
    parser.add_argument('--epochs', '-e', help='number of epochs', default=30)
    parser.add_argument('--pretrained', help="Use pretrained model", action='store_true')
    args = parser.parse_args()

    model = CifarPytorchTrainer(args.model_name, use_existing_model=args.pretrained, lr=args.lr, epochs=args.epochs)
    print(model.model)
    if not args.pretrained:
        print(
            f'Training {args.model_name} with lr of {args.lr} and {args.epochs} epochs. Results will be saved to {args.output_json_path}')
        model.train()
    else:
        print(f'Using {args.model_name} pretrained model. Results will be saved to {args.output_json_path}')
    model.test()
    model.visualise()
    model.save()
