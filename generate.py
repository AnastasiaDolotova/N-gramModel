from train import Model, argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='путь к файлу, из которого загружается модель.')
parser.add_argument('--prefix', type=str, help='необязательный аргумент. Начало предложения (одно или несколько слов)')
parser.add_argument('--length', type=int, help='длина генерируемой последовательности.')

args = parser.parse_args()
model = Model()
model.load_model(args.model)

print(model.generate(args.length, args.prefix))
