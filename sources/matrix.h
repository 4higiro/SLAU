#ifndef MATRIX // Защита от повторного включения
#define MATRIX

typedef vector<double> vecd;			// Массив дробных чисел - вектор
typedef vector<vector<double>> matd;	// Двумерный массив дробных чисел - матрица

// Класс матрицы элементов double
class matrixd
{
private:
	matd arr; // Матрица
public:
	// Конвертация типов матриц из библиотеки OpenCV в объект класса matrixd
	static matrixd cvtCVMat2NNMat(Mat& img)
	{
		// img - матрица библиотеки OpenCV
		uchar* matrix = img.ptr(); // Извлечение данных изображения
		matrixd result; // Создание объекта класса matrixd
		result.resize(MAT_HEIGHT, MAT_WIDTH); // Размером соответствующим распознаваемому изображению
		for (int i = 0; i < MAT_HEIGHT; i++)
		{
			for (int j = 0; j < MAT_WIDTH; j++)
			{
				// Запись извлеченных данных
				if ((int)matrix[i * MAT_WIDTH + j] > 100)
					result[i][j] = 1.0;
				else
					result[i][j] = 0.0;
			}
		}
		return result; // Возврат результата
	}

	// Конструктор по умолчанию
	matrixd() { }

	// Выделение памяти
	void resize(uint rows, uint cols)
	{
		// rows - кол-во строк
		// cols - кол-во столбцов

		// Очистка старой матрицы
		for (int i = 0; i < arr.size(); i++)
			arr[i].clear();
		arr.clear();

		// Создание новой (все элементы 0.0)
		arr.resize(rows);
		for (int i = 0; i < rows; i++)
			arr[i].resize(cols);
	}

	// Оператор индексации
	vecd& operator[](int index)
	{
		return arr[index];
	}

	// Геттер кол-ва строк
	uint rCount()
	{
		return arr.size();
	}

	// Геттер кол-ва столбцов
	uint cCount()
	{
		// Если в матрице нет ни одной строки - кол-во столбцов равно 0
		if (rCount() != 0)
			return arr[0].size();
		else
			return 0;
	}

	// Геттер данных, хранящихся в матрице
	double** pointer()
	{
		double** matrix = new double*[rCount()]; // Создание матрицы
		for (int i = 0; i < rCount(); i++)
		{
			matrix[i] = new double[cCount()];
			for (int j = 0; j < cCount(); j++)
				matrix[i][j] = arr[i][j]; // Заполнение хранящимися данными
		}
		return matrix; // Возврат результата
	}

	// Изменение кол-ва знаков после запятой у каждого элемента матрицы
	void setSign(uint sign_count)
	{
		// sign_count - кол-во знаков
		for (int i = 0; i < rCount(); i++)
		{
			for (int j = 0; j < cCount(); j++)
			{
				// Оставляется только целая часть числа с нужным кол-во знаков
				int number = arr[i][j] * (double)pow(10.0f, (double)sign_count);
				arr[i][j] = number / (double)pow(10.0f, (double)sign_count);
			}
		}
	}

	// Заполнение матрицы случайными числами в нужном диапазоне с нужным кол-вом знаков
	void setRand(int min_value, int max_value, uint sign_count = 0)
	{
		// min_value, max_value - диапазон
		// sign_count - кол-во знаков после запятой
		if (max_value < min_value)
			throw MATRIX_UNCORRECT_RAND_GEN; // Некорректный диапазон (если нижняя граница выще верхней)

		// Заполнение случаными числами
		int diff = max_value - min_value;
		for (int i = 0; i < rCount(); i++)
		{
			for (int j = 0; j < cCount(); j++)
				arr[i][j] = (rand() % (diff * (int)pow(10, (int)sign_count)) - ((diff / 2) * (int)pow(10, (int)sign_count))) / (double)(pow(10.0f, (double)sign_count));
		}
	}

	// Матрично-векторное умножение
	vecd operator*(vector<double> v)
	{
		// v - преобразуемый вектор
		if (v.size() != cCount())
			throw MATRIX_UNCORRECT_MULT_VECTOR; // Размер вектора должен совпадать с кол-во столбцов матрицы

		vecd result; // Преобразованный вектор
		result.resize(rCount()); // Размер полученного вектора будет равен кол-во строк матрицы
		for (int i = 0; i < rCount(); i++)
		{
			for (int j = 0; j < cCount(); j++)
				result[i] += arr[i][j] * v[j]; // Преобразование
		}

		return result; // Возврат результата
	}

	// Матрично-скалярное умножение
	matrixd operator*(double scalar)
	{
		// scalar - константа, масштабирующая вектор-стобцы матрицы
		matrixd result = *this; // Результрующая матрицы
		for (int i = 0; i < rCount(); i++)
		{
			for (int j = 0; j < cCount(); j++)
				result[i][j] = arr[i][j] * scalar; // Умножение
		}
		return result; // Возврат результата
	}
};

#endif
