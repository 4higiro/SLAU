#ifndef NURO // Защита от повторного включения
#define NURO

#include "slau.h" // Включение всех заголовочных файлов

typedef vector<matrixd> w_mats;			// Скоращенное название структуры хранения весовых коэф-тов
typedef vector<vector<double>> b_vecs;	// Сокращенное название структуры хранения смещений
typedef vector<uint> d_vec;				// Скоращенное название структуры хранения размерностей векторов нейронов

// Посчитанные дельты весов и смещений перцептрона
struct differential
{
	w_mats d_weights;	// Дельты весов
	b_vecs d_biases;	// Дельты смещений

	// Выделение памяти под дельты
	void resize(d_vec& dims)
	{
		// dims - размерности векторов нейронов
		d_weights.resize(dims.size() - 1);
		for (int i = 0; i < d_weights.size(); i++)
			d_weights[i].resize(dims[i + 1], dims[i]);
		d_biases.resize(dims.size() - 1);
		for (int i = 0; i < d_biases.size(); i++)
			d_biases[i].resize(dims[i + 1]);
	}
};

// Полносвязная нейросеть
class perceptron
{
private:
	w_mats weights;			// Весовые коэф-ты
	b_vecs biases;			// Смещения
	d_vec dims;				// Размерности векторов нейронов
	differential last;		// Дифференциал, посчитанный на предыдущей итерации обучения
	vector<vecd> nuerals;	// Вектора нейронов

	// Сигмодида
	static double sigm(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	perceptron() {}; // Конструктор по умолчанию

	friend class cnn; // Дружественен для класса сверточной нейросети
public:
	double learning_rate = 0.9;	// Коэфициент спуска по градиенту
	double moment = 0.1;		// Коэфициент инерции спуска

	// Конструктор
	perceptron(d_vec& dims)
	{
		// dims - кол-во нейронов на каждом слое
		weights.resize(dims.size() - 1);	// Создание матриц весовых коэф-тов
		for (int i = 0; i < weights.size(); i++)
		{
			weights[i].resize(dims[i + 1], dims[i]);
			weights[i].setRand(-1, 1, 1);
		}
		biases.resize(dims.size() - 1);		// Создание векторов смещений
		for (int i = 0; i < biases.size(); i++)
		{
			biases[i].resize(dims[i + 1]);
			for (int j = 0; j < biases.size(); j++)
				biases[i][j] = (rand() % 20 - 10) / 10.0;
		}
		last.resize(dims);				// Создание диффиренциала последней итерации		
		nuerals.resize(dims.size());	// Создание векторов нейронов
		this->dims = d_vec(dims);		// Копирование размерностей векторов нейронов
	}

	// Конвертер матрицы изображения в вектор
	static vecd cvtImgMat2Vec(matrixd& img)
	{
		// img - входное изображение
		vecd result; // Перезапись всех строк подряд в одну
		result.resize(img.rCount() * img.cCount());
		for (int i = 0; i < img.rCount(); i++)
		{
			for (int j = 0; j < img.cCount(); j++)
				result[i * img.cCount() + j] = img[i][j];
		}
		return result; // Возврат вектора значений изображения
	}

	// Конвертер классового идентификатора в целевой вектор
	static vecd cvtAnswer2Vec(uint answer)
	{
		// answer - идентификатор распознанного символа
		if (answer >= SYM_COUNT)
			throw NURO_UNCORRECT_CVT_ANSWER; // Если идентификатор не существует - генерация исключения

		vecd result; // Создание 0-вектора
		result.resize(SYM_COUNT);
		result[answer] = 1.0; // На месте соответствующего нейрона максимальная степень активации
		return result; // Возврат целевого вектора
	}

	// Конвертер классового идентификатора в распознанный символ
	static string cvtAnswer(uint answer)
	{
		// answer - идентификатор распознанного символа
		if (answer >= SYM_COUNT)
			throw NURO_UNCORRECT_CVT_ANSWER; // Если идентификатор не существует - генерация исключения

		// Все символы (порядковый номер равен классовому идентификатору)
		string syms[SYM_COUNT] = {
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"a", "b", "c", "d", "e", "f", "g", "m", "n", "p",
			"q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
			"pi", "+", "-", "="
		};

		return syms[answer]; // Возврат распознанного символа
	}

	// Конвертер cимвола в классовый идентификатор
	static uint cvtAnswer(string answer)
	{
		// answer - размеченный символ

		// Все символы (порядковый номер равен классовому идентификатору)
		string syms[SYM_COUNT] = {
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"a", "b", "c", "d", "e", "f", "g", "m", "n", "p",
			"q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
			"pi", "+", "-", "="
		};

		// Поиск индекса искомого символа
		uint result = SYM_COUNT;
		for (int i = 0; i < SYM_COUNT; i++)
		{
			if (answer == syms[i])
			{
				result = i;
				break;
			}
		}

		if (result == SYM_COUNT)
			throw NURO_UNCORRECT_CVT_ANSWER; // Если идентификатор не существует - генерация исключения

		return result; // Возврат результат
	}

	// Определение класса принадлежности изображения
	static uint softMax(vecd& nuerals_rs)
	{
		// nuerals_rs - выходной вектор нейросети
		uint m = 0;
		for (int i = 0; i < nuerals_rs.size(); i++)
		{
			if (nuerals_rs[i] > nuerals_rs[m])
				m = i; // Поиск нейрона с максимальной активацией
		}
		return m; // Порядковый номер такого нейрона - классовый идентификатор распозноваемого символа
	}

	// Печать обучаемых параметров сети на экран
	void printWeights()
	{
		for (int i = 0; i < weights.size(); i++)
		{
			cout << "MAT " << i << ":" << endl;
			for (int j = 0; j < weights[i].rCount(); j++)
			{
				for (int k = 0; k < weights[i].cCount(); k++)
					cout << weights[i][j][k] << "  ";
				cout << endl;
			}
			cout << "//// BIASES:" << endl;
			for (int j = 0; j < biases[i].size(); j++)
				cout << biases[i][j] << "  ";
			cout << endl << endl << endl;
		}
	}

	// Запись обучаемых параметров сети в файл
	void fprintWeights(string path)
	{
		ofstream fout;
		fout.open(path);
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].rCount(); j++)
			{
				for (int k = 0; k < weights[i].cCount(); k++)
					fout << weights[i][j][k] << "  ";
				fout << endl;
			}
			for (int j = 0; j < biases[i].size(); j++)
				fout << biases[i][j] << "  ";
			fout << endl << endl << endl;
		}
		fout.close();
	}

	// Чтение обучаемых параметров сети из файла
	void fscanWeights(string path)
	{
		ifstream fin;
		fin.open(path);
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].rCount(); j++)
			{
				for (int k = 0; k < weights[i].cCount(); k++)
					fin >> weights[i][j][k];
			}
			for (int j = 0; j < biases[i].size(); j++)
				fin >> biases[i][j];
		}
		fin.close();
	}

	// Определение значений выходного вектора нейросети (прямое распространение)
	vecd forwardPropagetion(vecd& input)
	{
		// input - вектор входных значений
		vecd current = vecd(input); // Промежуточный вектор значений функции (изначально равен входному)
		nuerals[0] = vecd(current); // Запись вектора в структуру хранения значений векторов нейронов
		for (int i = 0; i < weights.size(); i++)
		{
			current = weights[i] * current; // Линейное преобразование
			for (int j = 0; j < current.size(); j++)
			{
				current[j] = perceptron::sigm(current[j] + biases[i][j]); // Нелинейное преобразование
				// Округление до необходимого знака после запятой
				int number = current[j] * (double)pow(10.0f, (double)NURO_SIGN_COUNT);
				current[j] = number / (double)pow(10.0f, (double)NURO_SIGN_COUNT);
			}
			nuerals[i + 1] = vecd(current); // Запись вектора в структуру хранения значений векторов нейронов
		}
		return current; // После всех преобразований current - выходной вектор
	}

	// Подсчет градиента нейросети (обратное распространение)
	differential backPropagetion(vecd& input, vecd& pred)
	{
		// input - вектор входных значений 
		// pred - целевой вектор
		vecd output = forwardPropagetion(input); // Прямое распространение (получение значений нейронов на каждом слое)
		differential dnuro; // Создание структуры хранения дифференциала
		dnuro.resize(dims);
		
		// Подсчет частных производных по весам и смещениям на последнем слое нейросети
		for (int i = 0; i < weights[weights.size() - 1].rCount(); i++)
		{
			for (int j = 0; j < weights[weights.size() - 1].cCount(); j++)
				dnuro.d_weights[weights.size() - 1][i][j] = 2.0 * (output[i] - pred[i]) * output[i] * (1.0 - output[i]) * nuerals[nuerals.size() - 2][j];
			dnuro.d_biases[weights.size() - 1][i] = 2.0 * (output[i] - pred[i]) * output[i] * (1.0 - output[i]);
		}

		// Остальные производные вырожаются реккурентно от последнего слоя к первому
		for (int i = weights.size() - 2; i >= 0; i--)
		{
			for (int j = 0; j < weights[i].rCount(); j++)
			{
				double sum_errs = 0.0;
				for (int k = 0; k < weights[i + 1].rCount(); k++)
					sum_errs += weights[i + 1][k][j] * dnuro.d_weights[i + 1][k][j];
				for (int k = 0; k < weights[i].cCount(); k++)
					dnuro.d_weights[i][j][k] = sum_errs * (1.0 - nuerals[i + 1][j]) * nuerals[i][k];
				dnuro.d_biases[i][j] = sum_errs * (1.0 - nuerals[i + 1][j]);
			}
		}

		return dnuro; // Возврат посчитанного дифференциала
	}

	// Градиентный спуск (изменения весов и смещений)
	void learn(differential& dnuro)
	{
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].rCount(); j++)
			{
				for (int k = 0; k < weights[i].cCount(); k++)
				{
					weights[i][j][k] -= learning_rate * dnuro.d_weights[i][j][k] + moment * last.d_weights[i][j][k];
					last.d_weights[i][j][k] = learning_rate * dnuro.d_weights[i][j][k] + moment * last.d_weights[i][j][k];
				}
				biases[i][j] -= learning_rate * dnuro.d_biases[i][j] + moment * last.d_biases[i][j];
				int number = biases[i][j] * (double)pow(10.0f, (double)NURO_SIGN_COUNT);
				biases[i][j] = number / (double)pow(10.0f, (double)NURO_SIGN_COUNT);
				last.d_biases[i][j] = learning_rate * dnuro.d_biases[i][j] + moment * last.d_biases[i][j];
			}
			weights[i].setSign(NURO_SIGN_COUNT);
		}
	}
};

#endif
