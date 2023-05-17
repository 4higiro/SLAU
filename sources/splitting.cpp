#include "slau.h" // Включение всех заголовочных файлов

// Создание разбиения
vector<Mat> splitting(Mat& img, uint img_width, uint img_height, uint sp_width, uint sp_height)
{
	// img_width  - ширина исходного изображения
	// img_height - высота исходного ихображения
	// sp_width   - ширина клетки разбиения
	// sp_height  - высота клетки разбиения
	if (img_height % sp_height != 0 || img_width % sp_width != 0)
		throw SPLIT_ERROR; // Если размеры изображения не кратны размеру сплита - генереция исключения
	if (img.size().height != img_height || img.size().width != img_width)
		throw SPLIT_ERROR; // Если размер изображения не совпадает с размерностью матрицы - генерация исключения

	vector<Mat> split; // Разбиение
	for (int j = 0; j < img_height; j += sp_height)
	{
		for (int i = 0; i < img_width; i += sp_width)
		{
			// Обрезание исходного изображение по заданным параметрам разбиения
			cv::Rect roi(Point(i, j), Point(i + sp_width, j + sp_height));
			Mat split_rs, temp = img(roi);
			temp.copyTo(split_rs);
			split.push_back(split_rs);
		}
	}
	return split; // Возврат результата
}

// Сбор разбиения в единое изображение
Mat unsplitting(vector<Mat>& split, uint img_width, uint img_height, uint sp_width, uint sp_height)
{
	// img_width  - ширина исходного изображения
	// img_height - высота исходного ихображения
	// sp_width   - ширина клетки разбиения
	// sp_height  - высота клетки разбиения
	if (img_height % sp_height != 0 || img_width % sp_width != 0)
		throw SPLIT_ERROR; // Если размеры изображения не кратны размеру сплита - генереция исключения
	if (split.size() == 0)
		throw SPLIT_ERROR; // Если вектор сплитов пуст - генерация исключения
	if (split[0].size().height != sp_height || split[0].size().width != sp_width)
		throw SPLIT_ERROR; // Если размер сплита не совпадает с размерностью матрицы - генерация исключения

	Mat result(Size(img_width, img_height), CV_8UC1, Scalar(0, 0, 0)); // Создание результирующего изображения
	uchar* matrix = result.ptr(); // Матрица результирующего изображения
	for (int i = 0; i < img_height / sp_height; i++)
	{
		for (int j = 0; j < img_width / sp_width; j++)
		{
			uchar* temp = split[i * (img_width / sp_width) + j].ptr(); // Матрица изображения из разбиения
			for (int x = 0; x < sp_height; x++)
			{
				for (int y = 0; y < sp_width; y++)
					matrix[(i * sp_height + x) * img_width + j * sp_width + y] = temp[x * sp_width + y];
				// Данные всех изображений из разбиения встают на свои места в результирующее изображение
			}
		}
	}
	return result; // Возврат результата
}