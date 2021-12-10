import fasttext.util
import warnings
warnings.filterwarnings('ignore')


# model = fasttext.load_model('D:\cc.de.300.bin')

model = fasttext.train_supervised('FastText_input.txt',wordNgrams = 2)

terms = input("Give two terms with space between them: ")
print(type(model.predict(terms)[0]))
print(type(model.predict(terms)[1]))
print(model.predict(terms))