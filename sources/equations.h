#ifndef EQUATIONS // Защита от повторного включения
#define EQUATIONS

#include "slau.h" // Включение всех заголовочных файлов

// Абстрактный класс СЛАУ
class equations
{
protected:
	matrixd koeffs;				// Матрица коэффициентов
	vector<double> free_terms;	// Вектор свободных членов
	uint n;						// Кол-во переменных

	static double determinant(matrixd matrix, uint n)
	{
		// matrix - матрица
		// n - размерность матрицы
		if (n <= 1)
			return matrix[0][0]; // Определитель матрицы размерности 1
		else if (n == 2)
			return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]); // Определитель матрицы размерности 2
		else
		{
			// Определитель матрицы размерности 3 и более
			double result = 0; // Результат подсчета (изначально 0)
			for (int k = 0; k < n; k++) // Разложение по строке
			{
				// Создание матрицы минора
				matrixd minor;
				minor.resize(n - 1, n - 1);
				for (int i = 0; i < n - 1; i++)
				{
					bool column = false;
					for (int j = 0; j < n - 1; j++)
					{
						if (j == k) // Исключение k-го столбца из минора
							column = true;
						// Исключается также 0-я строка из минора
						if (column)
							minor[i][j] = matrix[i + 1][j + 1];
						else
							minor[i][j] = matrix[i + 1][j];
					}
				}
				// Подсчет определителя (рекурсивно)
				result += (double)pow(-1.0, k) * determinant(minor, n - 1) * matrix[0][k];
			}
			return result; // Возврат результата
		}
	}
	
	// Выделение памяти для СЛАУ
	void setSystemOfEq()
	{
		koeffs.resize(n, n);
		free_terms.resize(n);
	}

public:
	// Решение СЛАУ
	virtual vector<double> getRoots() = 0;

	// Заполнение поля матрицы коэфициентов принимаемой матрицей
	void fillMatrix(matrixd& matrix)
	{
		// matrix - матрица коэффицентов
		koeffs = matrix;
	}

	// Заполнение поля массива свободных членов принимаемым массивом
	void fillFreeTerms(vector<double> free_terms)
	{
		// free_terms - вектор свободных членов
		this->free_terms = vector<double>(free_terms);
	}
};

// Класс решения СЛАУ методом крамера
class kramer : public equations
{
public:
	// Конструктор для метода Крамера
	kramer(uint n)
	{
		// n - размерность системы
		this->n = n;
		setSystemOfEq();
	}

	// Решение СЛАУ методом Крамера
	vector<double> getRoots() override
	{
		double delta = determinant(koeffs, n); // Подсчет определителя
		if (delta == 0.0)
			throw INFINITY_SET_OF_SOLUTIONS; // Если определитель равен 0 - бесконечное множество решений

		// Вектор корней системы
		vector<double> roots;
		roots.resize(n);

		// Подсчет всех значений дельты по формуле Крамера
		for (int k = 0; k < n; k++)
		{
			matrixd delta_matrix;
			delta_matrix.resize(n, n);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					// Заполнение k-го столбца матрицы вектором свободных членов
					if (j == k)
						delta_matrix[i][j] = free_terms[i];
					else
						delta_matrix[i][j] = koeffs[i][j];
				}
			}
			// Подсчет корней
			roots[k] = (double)determinant(delta_matrix, n) / delta;
		}

		return roots; // Возврат результата
	}
};

// Класс решения СЛАУ методом Гаусса
class gauss : public equations
{
public:
	// Конструктор для метода Гаусса
	gauss(uint n)
	{
		// n - размерность системы
		this->n = n;
		setSystemOfEq();
	}

	// Решение СЛАУ методом Гаусса
	vector<double> getRoots() override
	{
		double delta = determinant(koeffs, n); // Подсчет определителя
		if (delta == 0.0)
			throw INFINITY_SET_OF_SOLUTIONS; // Если определитель равен 0 - бесконечное множество решений

		// Вектор корней системы
		vector<double> roots;
		roots.resize(n);

		// Создание расширенной матрицы
		matrixd full_matrix;
		full_matrix.resize(n, n + 1);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				full_matrix[i][j] = koeffs[i][j];
			full_matrix[i][n] = free_terms[i]; // Последний вектор-столбец - свободные члены
		}

		// Элементарные преобразования расширенной матрицы
		for (int i = 0; i < n - 1; i++)
		{
			// Деление строки на диагональный коэфициент
			double temp = full_matrix[i][i];
			for (int j = i; j < n + 1; j++)
				full_matrix[i][j] /= temp;
			// Вычитание строк матрицы
			for (int k = i + 1; k < n; k++)
			{
				temp = full_matrix[k][i];
				for (int j = i; j < n + 1; j++)
					full_matrix[k][j] -= temp * full_matrix[i][j];
			}
		}
		// В результате в нижнем треугольнике матрицы - все 0

		// Обратный ход метода Гаусса
		double temp = full_matrix[n - 1][n - 1];
		for (int j = n - 1; j < n + 1; j++)
			full_matrix[n - 1][j] /= temp; // Делим последнюю строку на диагональный элемент
		// Подсчет корней
		roots[n - 1] = full_matrix[n - 1][n]; // Считаем последний корень
		for (int i = n - 2; i >= 0; i--)
		{
			// Остальные корни вырожаем через последний
			roots[i] = full_matrix[i][n];
			for (int j = i + 1; j < n; j++)
				roots[i] -= full_matrix[i][j] * roots[j]; // Подсчет корней
		}

		return roots; // Возврат результата
	}
};

// Класс решения СЛАУ методом Жордана-Гаусса
class jordan_gauss : public equations
{
public:
	// Конструктор для метода Жордана-Гаусса
	jordan_gauss(uint n)
	{
		// n - размерность системы
		this->n = n;
		setSystemOfEq();
	}

	// Решение СЛАУ методом Жордана-Гаусса
	vector<double> getRoots() override
	{
		double delta = determinant(koeffs, n); // Подсчет определителя
		if (delta == 0.0)
			throw INFINITY_SET_OF_SOLUTIONS; // Если определитель равен 0 - бесконечное множество решений

		// Вектор корней системы
		vector<double> roots;
		roots.resize(n);

		// Создание расширенной матрицы
		matrixd full_matrix;
		full_matrix.resize(n, n + 1);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				full_matrix[i][j] = koeffs[i][j];
			full_matrix[i][n] = free_terms[i]; // Последний вектор-столбец - свободные члены
		}

		// Элементарные преобразования расширенной матрицы
		for (int i = 0; i < n - 1; i++)
		{
			// Деление строки на диагональный элемент
			double temp = full_matrix[i][i];
			for (int j = i; j < n + 1; j++)
				full_matrix[i][j] /= temp;
			// Вычитание строк матрицы
			for (int k = i + 1; k < n; k++)
			{
				temp = full_matrix[k][i];
				for (int j = i; j < n + 1; j++)
					full_matrix[k][j] -= temp * full_matrix[i][j];
			}
		}

		// Обратный ход (также через элементарные преобразования)
		for (int i = n - 1; i > 0; i--)
		{
			// Деление строки на диагональный элемент
			double temp = full_matrix[i][i];
			for (int j = i; j < n + 1; j++)
				full_matrix[i][j] /= temp;
			// Вычитание строк
			for (int k = i - 1; k >= 0; k--)
			{
				temp = full_matrix[k][i];
				for (int j = i; j < n + 1; j++)
					full_matrix[k][j] -= temp * full_matrix[i][j];
			}
		}

		// Запись результатов вычислений
		for (int i = 0; i < n; i++)
			roots[i] = full_matrix[i][n];

		return roots; // Возврат результата
	}
};

// Класс решения СЛАУ методом Простой Итерации
class simple : public equations
{
public:
	// Конструктор для метода Простой Итерации
	simple(uint n)
	{
		// n - размерность системы
		this->n = n;
		setSystemOfEq();
	}

	// Решение СЛАУ методом Простой Итерации
	vector<double> getRoots() override
	{
		double delta = determinant(koeffs, n); // Подсчет определителя
		if (delta == 0.0)
			throw INFINITY_SET_OF_SOLUTIONS; // Если определитель равен 0 - бесконечное множество решений

		// Вектор корней системы
		vector<double> roots;
		roots.resize(n);

		// Проверка условия сходимости приближений
		for (int i = 0; i < n; i++)
		{
			double sum = 0;
			for (int j = 0; j < n; j++)
			{
				if (i != j)
					sum += abs(koeffs[i][j]);
			}
			// Если сумма модулей коэфициентов строки матрицы меньше модуля диагонального элемента
			if (sum > abs(koeffs[i][i]))
				throw OUT_CONVERGENCE; // Нет сходимости
		}

		// Вычисление нулевого приближения
		for (int i = 0; i < n; i++)
			roots[i] = free_terms[i] / koeffs[i][i];

		// Вычисление остальных приближений (всего 100)
		for (int k = 0; k < 100; k++)
		{
			for (int i = 0; i < n; i++)
			{
				// Запись предыдущего приближения
				vector<double> last;
				last.resize(n);
				for (int j = 0; j < n; j++)
					last[j] = roots[j];
				// Подсчет следующего приближения
				double sum = 0.0;
				for (int j = 0; j < n; j++)
				{
					if (i != j)
						sum -= koeffs[i][j] * last[j];
				}
				sum += free_terms[i];
				roots[i] = sum / koeffs[i][i]; // Вычисление корней
			}
		}

		return roots; // Возврат результата
	}
};

// Класс решения СЛАУ методом Зейделя
class zeydel : public equations
{
public:
	// Конструктор для метода Зейделя
	zeydel(uint n)
	{
		// n - размерность системы
		this->n = n;
		setSystemOfEq();
	}

	// Решение СЛАУ методом Зейделя
	vector<double> getRoots() override
	{
		double delta = determinant(koeffs, n); // Подсчет определителя
		if (delta == 0.0)
			throw INFINITY_SET_OF_SOLUTIONS; // Если определитель равен 0 - бесконечное множество решений

		// Вектор корней системы
		vector<double> roots;
		roots.resize(n);

		// Проверка условия сходимости приближений
		for (int i = 0; i < n; i++)
		{
			double sum = 0;
			for (int j = 0; j < n; j++)
			{
				if (i != j)
					sum += abs(koeffs[i][j]);
			}
			// Если сумма модулей коэфициентов строки матрицы меньше модуля диагонального элемента
			if (sum > abs(koeffs[i][i]))
				throw OUT_CONVERGENCE; // Нет сходимости
		}

		// Вычисление нулевого приближения
		for (int i = 0; i < n; i++)
			roots[i] = free_terms[i] / koeffs[i][i];

		// Вычисление остальных приближений (всего 10)
		for (int k = 0; k < 10; k++)
		{
			for (int i = 0; i < n; i++)
			{
				// Подсчет следующего приближения
				double sum = 0.0;
				for (int j = 0; j < n; j++)
				{
					// Отсутсвует запись предыдущего приближения
					// Таким образом в методе Зейделя уже вычисленные приближения
					// используются в процессе вычислений текущих приближений
					if (i != j)
						sum -= koeffs[i][j] * roots[j];
				}
				sum += free_terms[i];
				roots[i] = sum / koeffs[i][i]; // Вычисление корней
			}
		}

		return roots; // Возврат результата
	}
};

#endif