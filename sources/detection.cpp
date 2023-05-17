#include "slau.h" // Включение всех заголовочных файлов

// Подсчет средней яркости изображения
int calcAvgBnss(Mat& current)
{
	int avgBnss = 0; // Средняя яркость
	for (int i = 0; i < current.size().height; i++)
	{
		long long sumBnss = 0;
		for (int j = 0; j < current.size().width; j++)
			sumBnss += (int)current.at<Vec3b>(i, j)[0];
		avgBnss += sumBnss / current.size().width;
	}
	avgBnss /= current.size().height;
	return avgBnss; // Возврат результата
}

//	Предобработка изображения СЛАУ
Mat preproccesing(Mat& img, int color, uint(*approx)(uint x, int color))
{
	/*
		В предобработке необходимо выделить белым цветом важные для распознования детали белым цветов,
		а остальные закрасить черным (img-->current).

		Для этого необходимо избавиться от фактора цвета чернил и фона, главное чтобы символы были 
		оттчетливо видны и изображены более темными оттенками на более светлом фоне.
		Чтобы этого добиться необходимо изображение в одноканальном цветовом пространстве (черно-белое),
		в зависимости от средней яркости изображения, выделить рукописные символы. Для этого
		необходимо перевести изображение в цветовое пространство HSV и определить диапазон насыщенности цвета
		важных для нас структурных элементов (рукописных символов).
	*/

	// approx - аппроксимационная функция зависимости параметра vmax от средней яркости

	// Копирование исходного изображения в рабочую матрицу
	Mat current;
	img.copyTo(current);

	// Подсчет средней яркости пикселей изображения
	int avg_bnss = calcAvgBnss(current);

	// Перевод изображения в цветовое пространство HSV (GRAY-->BGR-->HSV)
	cvtColor(current, current, COLOR_GRAY2BGR);
	cvtColor(current, current, COLOR_BGR2HSV);

	// Диапазон цвета HSV для выделяемых структурных элементов
	int hmin = 0, hmax = 179;
	int smin = 0, smax = 255;
	int vmin = 0, vmax = approx(avg_bnss, color);

	// Выделение рукописных символов
	inRange(current, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), current);
	Mat ker = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(current, current, ker);

	return current; // Возврат предобработноого изображения
}

// Очистка шумов
Mat noiseClear(Mat& current, int lim)
{
	// lim - порог площади контура шума
	Mat rs(current.size(), CV_8UC1, Scalar(0, 0, 0)); // Создание пустого одноканального изображения

	// Обнаружение контуров средствами OpenCV
	vector<vector<Point>> ctrs;
	vector<Vec4i> rar;
	findContours(current, ctrs, rar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Исключение шумов
	for (int i = 0; i < ctrs.size(); i++)
	{
		// Если площадь контура превышает пороговое значение - это не шум
		int area = contourArea(ctrs[i]);
		if (area > lim)
			drawContours(rs, ctrs, i, Scalar(255, 255, 255)); // Рендер контуров, не являющихся шумом
	}

	return rs; // Возврат результата
}

// Очистка фигурной скобки в массиве контуров
void clearBrace(vector<vector<Point>>& ctrs)
{
	// Отсечение фигурной скобки (она мешает, т.к. в ее контуре есть точки, 
	// находящиеся на одном уровне по у с каждым уравнением системы)
	int m = 0, value = 0; // Поиск контура с максимальной высотой
	float avg_value = 0;
	for (int i = 0; i < ctrs.size(); i++)
	{
		int min_point = 0; // Поиск точки с минимальной высотой
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].y < ctrs[i][min_point].y)
				min_point = j;
		}
		int max_point = 0; // Поиск точки с максимальной высотой
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].y > ctrs[i][max_point].y)
				max_point = j;
		}
		// Разница у-координаты этих точек есть высота контура
		if (ctrs[i][max_point].y - ctrs[i][min_point].y > value)
		{
			value = ctrs[i][max_point].y - ctrs[i][min_point].y;
			avg_value += value;
			m = i; // Записть контура с максимальной выостой
		}
	}

	// Очистка контура фигурной скобки
	avg_value /= (float)ctrs.size();
	if (value > avg_value * 2)
		ctrs[m].clear(); // Если высота найденного конутра превышает две средних высоты - конутр считается фигурной скобкой
}

// Объединение близких контуров
void unionNearCtrs(vector<vector<Point>>& ctrs, bool(*comp)(Point x), bool(*skip)(double two_area, double avg_area))
{
	// comp - условие соединения контуров

	/*
		Контуры, расположенные близко друг к другу, могут быть одним структурным элементов,
		т.к. точность предобработки не идеальная. В некоторых случаях несколько разных
		структурных элементов может быть необходимо объеденить в один (например несколько
		символов, расположенных на одной горизонтали, бывает нужно объеденить в один
		структурный элемент - уравнение). В этой функции реализован обход и поэлементное
		сравнение всех контуров, обнаруженных на изображении. Условие при котором контуры
		будут объединяться может быть любым, в зависимости от задачи.
	*/

	// Подсчет средней площади контуров
	double avg_area = 0.0;
	uint counter = 0;
	for (int i = 0; i < ctrs.size(); i++)
	{
		if (ctrs[i].size() > 10 && ctrs[i].size() < 110)
		{
			avg_area += contourArea(ctrs[i]);
			counter++;
		}
	}
	avg_area /= static_cast<double>(counter);

	// Обход всех контуров
	for (int i = 0; i < ctrs.size(); i++)
	{
		for (int j = 0; j < ctrs.size(); j++)
		{
			if (i == j || ctrs[i].size() < 10 || ctrs[j].size() < 10) // Одинаковые контуры друг с другом не сравниваются
				continue;
			// Контуры разных символов не объеденияются
			double two_area = contourArea(ctrs[i]) + contourArea(ctrs[j]);
			if (skip(two_area, avg_area))
				continue;
			// Обход всех точек каждого контуры
			for (int p = 0; p < ctrs[i].size(); p++)
			{
				for (int q = 0; q < ctrs[j].size(); q++)
				{
					// Вычисление вектора, проведенного от точки одного конутра к точке другого
					Point x = ctrs[i][p] - ctrs[j][q];
					if (comp(x)) // Определение истинности заданного условия
					{
						// Копирование всех точек одного контура в другой
						for (int k = 0; k < ctrs[j].size(); k++)
						{
							ctrs[i].push_back(ctrs[j][k]);
						}
						// Очистка скопированного контуры
						ctrs[j].clear();
						break;
					}
				}
			}
		}
	}
}

// Разделение изображения на структурные элементы
void fragmentation(Mat& current, vector<vector<Point>>& ctrs, vector<Mat>& elements, int pad_x, int pad_y)
{
	// elements		- массив изображений обнаруженных структурных элементов
	// pad_x, pad_y	- отступы от контуров

	// Обнаруженные и объединенные контуры на предобработанном изображении без шумов в этой
	// функции вырезаются в отдельное изображение и записываются в массив таких изображений

	elements.reserve(ctrs.size()); // Структурных элементов будет не больше чем обнаруженных контуров

	// Обход всех контуров
	for (int i = 0; i < ctrs.size(); i++)
	{
		// Если в контуре недостаточно точек - контур является шумом
		if (ctrs[i].size() < NOISE_POINTS_COUNT)
			continue;

		// Необходимо определить прямоугольник, в которй вписывается структурный элемент
		// Для этого нужно найти координаты двух точек, расположенные на диаганли такого прямоугольника

		int m1 = 0; // Поиск минимального значения х
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].x < ctrs[i][m1].x)
				m1 = j;
		}
		int m2 = 0; // Поиск максимально значения х
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].x > ctrs[i][m2].x)
				m2 = j;
		}
		int m3 = 0; // Поиск минимального значения у
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].y < ctrs[i][m3].y)
				m3 = j;
		}
		int m4 = 0; // Поиск максимального значения у
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].y > ctrs[i][m4].y)
				m4 = j;
		}

		// Определение точек на диаганали прямоугольника с отступом от контура
		Point x1 = Point(ctrs[i][m1].x - pad_x, ctrs[i][m3].y - pad_y); // Точка с минимальными найденными координатами
		Point x2 = Point(ctrs[i][m2].x + pad_x, ctrs[i][m4].y + pad_y); // Точка с максимальными найденными координатами

		cv::Rect roi(x1, x2); // Создание прямоугольника
		elements.push_back(current(roi)); // Обрезка и добавление изображения с найденным структурным элементом
	}
}

// Обнаружение уравнений
void equalsDetection(Mat& current, vector<vector<Point>>& ctrs, vector<Mat>& equals)
{
	// equals - массив изображений каждого уравнения из системы в отдельности

	/*
		Каждое уравнение из системы необходимо обрабатывать отдельно, т.к. в таком уравнении
		можно корректно задать порядок обнаружаемых символов (слева направо). Если на изображении
		присутствуют символы из других уравнений, то они не позволяют задать такой порядок. 
		Для этого необходимо объединить все найденные конутры с похожими значениями y 
		в один структурный элемент - уравнение, вырезав каждое такое уравнение в отдельное
		изображение.
	*/

	clearBrace(ctrs); // Удаление фигурной скобки

	// Объединение всех контуров, у-координаты точек которых не отличается друг от друга более чем на 5 пикселей
	unionNearCtrs(ctrs, [](Point x) {
		return sqrt(x.y * x.y) < 1;
		}, [](double a, double b) {
			return false; });

	// Обрезка изображение каждого уравнения и запись всех изображений уравнений в массив (отсутуп - 5 пикселей)
	fragmentation(current, ctrs, equals, 5, 5);
	std::reverse(begin(equals), end(equals));
}

// Сортировка массива обнаруженных символов в порядке записи уравнений
void symSort(vector<vector<Point>>& ctrs, vector<Mat>& syms)
{
	/*
		В каждом обнаруженном уравнении можно упорядочить обнаруженный символы
		в порядке слева-направо с помощью сравнения х-координаты их контуров
		(в данном случае сравнивается минимальная х-координата каждого контура)
	*/

	vector<int> points; // Массив минимальных х-координат
	points.reserve(syms.size()); // Он будет не больше чем массив обнаруженных символов
	// Обход всех контуров
	for (int i = 0; i < ctrs.size(); i++)
	{
		// Если в контуре недостаточно точек - контур является шумом
		if (ctrs[i].size() < NOISE_POINTS_COUNT)
			continue;

		int m1 = 0; // Поиск минимальной х-координаты
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].x < ctrs[i][m1].x)
				m1 = j;
		}

		points.push_back(ctrs[i][m1].x); // Запись в массив найденного значения
	}

	// Сортировка массива обнаруженных символов в соответсвтии с найденными точками методом простого обмена
	for (int i = 0; i < syms.size(); i++)
	{
		int m = 0;
		for (int j = 0; j < syms.size() - i; j++)
		{
			if (points[j] > points[m])
				m = j;
		}
		int temp_p = points[m];
		points[m] = points[points.size() - 1 - i];
		points[points.size() - 1 - i] = temp_p;
		Mat temp_m = syms[m];
		syms[m] = syms[syms.size() - 1 - i];
		syms[syms.size() - 1 - i] = temp_m;
	}
}

// Иллюстарция обнаружения
void detectDraw(Mat& img, vector<vector<Point>>& ctrs)
{
	// На исходном изображении выделяется зелеными прямоугольнками обнаруженные структурные элементы

	// Обход всех контуров
	for (int i = 0; i < ctrs.size(); i++)
	{
		// Если в контуре недостаточно точек - контур является шумом
		if (ctrs[i].size() < NOISE_POINTS_COUNT)
			continue;

		int m1 = 0; // Поиск минимального значения х
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].x < ctrs[i][m1].x)
				m1 = j;
		}
		int m2 = 0; // Поиск максимального значения х
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].x > ctrs[i][m2].x)
				m2 = j;
		}
		int m3 = 0; // Поиск минимального значения у
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].y < ctrs[i][m3].y)
				m3 = j;
		}
		int m4 = 0; // Поиск максимального значения у
		for (int j = 0; j < ctrs[i].size(); j++)
		{
			if (ctrs[i][j].y > ctrs[i][m4].y)
				m4 = j;
		}

		// Вычисление координат точек, находящихся на диаганали прямоугольника с отступом 5
		Point x1 = Point(ctrs[i][m1].x - 5, ctrs[i][m3].y - 5); // Точка с минимальными координатами
		Point x2 = Point(ctrs[i][m2].x + 5, ctrs[i][m4].y + 5); // Точка с максимлаьными коордтинатами

		rectangle(img, x1, x2, Scalar(0, 150, 0), 2); // Рендер прямоугольника
	}
}

// Получение исходных данных для распознования
void getSyms(Mat& img, vector<Mat>& syms)
{
	Mat current, rs;
	cvtColor(img, current, COLOR_BGR2GRAY); // Создание одноканального изображения
	resize(current, current, Size(IMG_WIDTH, IMG_HEIGHT));

	int color = calcAvgBnss(img); // Подсчет средней яркости изображения

	// Разбиение и предобработка
	vector<Mat> split = splitting(current, IMG_WIDTH, IMG_HEIGHT, 40, 40);
	for (int i = 0; i < split.size(); i++)
		split[i] = preproccesing(split[i], color, [](uint x, int color) {
		double b = 0.05 * color - 10;
		return static_cast<uint>(0.6 * x + b);
			});
	current = unsplitting(split, IMG_WIDTH, IMG_HEIGHT, 40, 40);

	rs = noiseClear(current, NOISE_AREA); // Очистка от шумов

	// Поиск и соединение всех близлежащих контуров
	vector<vector<Point>> ctrs;
	vector<Vec4i> rar;

	findContours(rs, ctrs, rar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	unionNearCtrs(ctrs, [](Point x) {
		return sqrt(x.x * x.x + x.y * x.y) < 15;
		}, [](double a, double b) {
			return a > 2.0 * b;
		});

	syms.reserve(ctrs.size()); // Символов будет не больше чем контуров

	// Построение общих структурных элементов для каждого уравнения из системы
	vector<Mat> equals;
	equalsDetection(current, ctrs, equals);

	// Обход всех уравнений
	for (int i = 0; i < equals.size(); i++)
	{
		// Добавление рамки вокруг изображения уравнения во избежание ошибок пэддинга
		rectangle(equals[i], cv::Rect(Point(0, 0), Point(equals[i].size().width, 2)), Scalar(0, 0, 0), 2);
		rectangle(equals[i], cv::Rect(Point(0, 0), Point(2, equals[i].size().height)), Scalar(0, 0, 0), 2);
		rectangle(equals[i], cv::Rect(Point(equals[i].size().width - 2, 0), Point(equals[i].size().width, equals[i].size().height)), Scalar(0, 0, 0), 2);
		rectangle(equals[i], cv::Rect(Point(0, equals[i].size().height - 2), Point(equals[i].size().width, equals[i].size().height)), Scalar(0, 0, 0), 2);

		Mat rs_equal = noiseClear(equals[i], NOISE_AREA); // Очистка уравнения от шумов
		ctrs.clear(); rar.clear(); // Очистка массивов с контурами
		findContours(rs_equal, ctrs, rar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // Поиск контуров на уравнении
		unionNearCtrs(ctrs, [](Point x) { // Объединение всез близлежащих контуров (чтобы каждый символ был единым контуром)
			return sqrt(x.x * x.x + x.y * x.y) < 15;
			}, [](double a, double b) {
				return a > 2.0 * b;
			});
		vector<Mat> eq_syms; // Массив всех обнаруженных символов в уравнении
		eq_syms.reserve(ctrs.size()); // Символов будет не больше чем контуров
		fragmentation(equals[i], ctrs, eq_syms, 3, 3); // Выделение каждого символа в отдельное изображение
		symSort(ctrs, eq_syms); // Сортировка (порядок слева-направо)
		for (int j = 0; j < eq_syms.size(); j++)
			syms.push_back(eq_syms[j]); // Копирование всех символов из уравнение в общий массив символов системы
		Mat temp(Size(MAT_WIDTH, MAT_HEIGHT), CV_8UC1, Scalar(0, 0, 0)); // Разделитель между уравнениями
		syms.push_back(temp); // Добавление разделителя
	}

	// Все изображения подгоняются по размеру для входного слоя нейросети
	for (int i = 0; i < syms.size(); i++)
		resize(syms[i], syms[i], Size(MAT_WIDTH, MAT_HEIGHT));
}

// Проверка изображение на наличие признаков разделителя
bool isSeparator(Mat& sym)
{
	// sym - изображение с символом
	vector<vector<Point>> ctrs;
	vector<Vec4i> rar;
	findContours(sym, ctrs, rar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	return ctrs.size() == 0; // Изображение, являющиеся разделителем не содержит контуров
}