#ifndef LEXER // Защита от повторного включения
#define LEXER

#include "slau.h" // Включение всех заголовочных файлов

// Типы лексим
enum token_type {
	TNUM, TVAR, TSIGN, TEND, TERR 
};

// Типы символов
enum sym_type {
	SNUM, SLET, SSIGN, SEND, SNONE
};

// Объектная оболочка для лексимы
class token
{
private:
	string value;		// Значение лексимы
	token_type type;	// Тип лексимы 

	// Конструкторы
	token() {}
	token(string _value, token_type _type) : value(_value), type(_type) {}

	friend class lexer; // Лексический анализатор имеет доступ к закрытым полям лексимы
};

// Класс лексического анализатора
class lexer
{
private:
	list<token> tokens; // Список лексим

	// Определение типа символа по таблице ASCII
	sym_type checkSym(char sym)
	{
		if (sym >= 48 && sym <= 57)
			return SNUM;
		else if (sym >= 97 && sym <= 122)
			return SLET;
		else if (sym == 45 || sym == 43 || sym == 61)
			return SSIGN;
		else if (sym == '\n')
			return SEND;
		else
			return SNONE;
	}
public:
	string str; // Анализируемая строка

	// Создание списка лексим (потоковый ввод строки)
	void tokenizing()
	{
		string value = "";		// Значение вводимой лексимы
		token_type type = TERR;	// Тип вводимой лексимы
		str += "\n";			// Ограничение на просмотр символов
		// Потоковый ввод
		for (int i = 0; i < str.length() - 1; i++)
		{
			if (str[i] == 'p' && str[i + 1] == 'i') // Ввод лексимы "Число pi"
			{
				i++;
				tokens.push_back(token("pi", TNUM));
				continue;
			}
			if (str[i] == 'e') // Ввод лексимы "Число e"
			{
				tokens.push_back(token("e", TNUM));
				continue;
			}
			for (i; checkSym(str[i]) == SNUM; i++)
			{
				value += str[i]; // Сбор символов числа в единую лексиму
				type = TNUM;
			}
			if (type == TNUM) // Ввод лексимы "Число"
			{
				for (int j = 0; j < value.length(); j++)
				{
					if (value[j] == '0')
					{
						value.erase(j, 1);
						j--;
					}
					else
						break;
				}
				tokens.push_back(token(value, type));
				value = "";
				type = TERR;
			}
			for (i; checkSym(str[i]) == SLET; i++)
			{
				if (str[i] == 'e' || (str[i] == 'p' && str[i + 1] == 'i')) // Символы e и pi это не переменные, а числа
				{
					i--;
					break;
				}
				value += str[i]; // Сбор символов переменной в единую лексиму
				type = TVAR;
			}
			if (type == TVAR) // Ввод лексимы "Переменная"
			{
				tokens.push_back(token(value, type));
				value = "";
				type = TERR;
			}
			if (checkSym(str[i]) == SSIGN) // Ввод лексимы "Знак"
			{
				value += str[i];
				type = TSIGN;
				tokens.push_back(token(value, type));
				value = "";
				type = TERR;
			}
			if (checkSym(str[i]) == SEND) // Ввод лексимы "Конец строки"
			{
				value += str[i];
				type = TEND;
				tokens.push_back(token(value, type));
				value = "";
				type = TERR;
			}
		}
	}

	// Корректировка ошибок распознавания
	void correction()
	{
		list<token> stream;				// Новый поток лексим
		uint eq_token_count = 0;		// Счетчки знаков "=" в уравнении
		token prev = token("", TERR);	// Предыдущая просматриваемая лексима
		double mult = 1.0;				// Несколько чисел подряд будут посчитаны как произведение
		bool is_double = false;			// Выведение десятичной части числа

		// Перебор всех лексим
		for (auto elem : tokens)
		{
			if (elem.value == "=") // Подсчет знаков "=" в уравнении
				eq_token_count++;
			if (eq_token_count > 1) // Если в уравнении уже есть знак "=", то он не вводится в поток
			{
				eq_token_count = 1;
				continue;
			}
			if (elem.type == TVAR && prev.type == TVAR) // Если в потоке подряд идут несколько переменных, то вводится только одна
				continue;
			if (elem.value == "+" && prev.value == "+") // Если в потоке подряд идут несколько знаков "+", то вводится только один
				continue;
			if (elem.value == "-" && prev.value == "-") // Если в потоке подряд идут несколько знаков "-", то вводится только один
				continue;
			if (elem.value == "-" && prev.value == "+") // Если в потоке подряд идут знаки "-" и "+", то вводится только "-"
			{
				stream.pop_back();
				elem.value = "-";
			}
			if (elem.value == "+" && prev.value == "-") // Если в потоке подряд идут знаки "+" и "-", то вводится только "-"
				continue;
			if (elem.type == TNUM && prev.type == TVAR) // Если в потоке число идет после переменной, то перед ней вводится знак "+"
				stream.push_back(token("+", TSIGN));
			if (elem.value == "pi" && elem.type == TNUM) // Произведение с числом п
			{
				mult *= 3.141592;
				prev = token(to_string(mult), TNUM);
				is_double = true;
				continue;
			}
			else if (elem.value == "e" && elem.type == TNUM) // Произведение с числом е
			{
				mult *= 2.718281;
				prev = token(to_string(mult), TNUM);
				is_double = true;
				continue;
			}
			else if (elem.type == TNUM) // Произведение с любым числом
			{
				if (is_double)
				{
					mult *= stoi(elem.value);
					prev = token(to_string(mult), TNUM);
				}
				else
					prev = elem;
				continue;
			}
			else if (prev.type == TNUM) // Ввод получившегося произведения
			{
				mult = 1.0;
				is_double = false;
				stream.push_back(prev);
			}
			if (elem.type == TEND && eq_token_count == 0) // Если в уравнении не было знака "=", то он добавляется в поток
			{
				stream.push_back(token("=", TSIGN));
				stream.push_back(token("0", TNUM));
			}
			if (elem.type == TEND) // Если дошли до конца строки, то обновляется счетчик знаков "=" в уравнении
				eq_token_count = 0;
			if (elem.value == "=" && (prev.type == TERR || prev.type == TEND)) // Если в начале уравнения вводится знак "=", то перед ним вводится число 0
				stream.push_back(token("0", TNUM));
			if (elem.type == TEND && prev.value == "=") // Если в конце уравнения вводится знак "=", то после него вводится число 0
				stream.push_back(token("0", TNUM));
			if (elem.type == TNUM && prev.value == "-") // Если в потоке вводится сначала знак "-", а затем число, то оба они вводятся как отрицательное число
			{
				prev = token("-" + elem.value, TNUM);
				stream.pop_back();
				stream.push_back(prev);
				continue;
			}
			if (elem.type == TVAR && prev.type != TNUM) // Если в потоке перед переменной не вводится число, то число 1 вводится перед переменной
				stream.push_back(token("1", TNUM));
			prev = elem; // Копирование текущего ввода в предыдущий
			stream.push_back(elem); // Потоковый ввод
		}
		tokens.clear(); // Очистка старого списка лексим
		tokens = list<token>(stream); // Перезапись списка лексим после окончания потокового ввода

		// Заполнение пустых численных токенов
		for (auto it = tokens.begin(); it != tokens.end(); it++)
		{
			if ((*it).type == TNUM && (*it).value == "")
				(*it).value = "0";
		}
	}

	// Запись всех лексим в строку
	string tok2Str()
	{
		string result = "";
		for (auto elem : tokens)
			result += elem.value;
		return result;
	}

	// Извлечение всех переменных из списка лексим
	vector<string> getVars()
	{
		vector<string> vars; // Переменные из системы
		vars.reserve(tokens.size()); // Кол-во переменных не больше чем кол-во лексим
		for (auto elem : tokens)
		{
			// Перебор всех лексим
			if (elem.type == TVAR)
				vars.push_back(elem.value); // Запись лексим "Переменная" в вектор переменных
		}
		// Оставляются только уникальный переменные
		sort(begin(vars), end(vars));
		vars.erase(unique(begin(vars), end(vars)), end(vars));
		return vars; // Возврат результата
	}

	// Извлечение матрицы коэффициентов и вектора свободных членов системы из списка лексим
	void exclude(matrixd& koeffs, vector<double>& free_terms)
	{
		// koeffs		- матрица для записи коэффициентов 
		// free_terms	- вектор для записи свободных членов
		vector<string> vars = getVars();	// Извлечение переменных из списка лексим
		uint dim = vars.size();				// Размерность системы
		koeffs.resize(dim, dim);			// Размерность матрицы совпадает с размерностью системы
		free_terms.resize(dim);				// Размерность вектора свободных членов совпадает с размерностью системы
		token prev = token("", TERR);		// Запиь предыдущей просмототренной лексимы
		bool side = false;					// false - левая часть уравнения, true - правая часть уравнения
		uint eq_count = 0;					// Счетчик уравнений системы

		// Пересчет кол-ва уравнений в системе
		for (auto elem : tokens)
		{
			if (elem.type == TEND)
				eq_count++;
		}

		// Генерация исключений при невозможности корректного извлечения данных
		if (dim == 0)
			throw UNCORRECT_NUMBER_OF_EQUALS; // Если в системе нет переменных
		if (eq_count > dim)
			throw UNCORRECT_NUMBER_OF_EQUALS; // Если в системе больше уравнений чем переменных

		eq_count = 0; // Обнуление счетчика уравнений

		// Перебор всех лексим
		for (auto elem : tokens)
		{
			if (!side && elem.type == TVAR && prev.type == TNUM)
			{
				// Коэффициенты перед одинаковыми перееменными из левой части уравнения суммируются
				for (int i = 0; i < dim; i++)
				{
					if (elem.value == vars[i]) // Сравнение переменной перед коэффициентом и его позиции в матрице
						koeffs[eq_count][i] += stod(prev.value);
				}
			}
			if (!side && elem.type != TVAR && prev.type == TNUM)
				free_terms[eq_count] -= stod(prev.value); // Свободные члены из левой части уравнения суммируются
			if (side && elem.type == TVAR && prev.type == TNUM)
			{
				// Коэффициенты перед одинаковыми перееменными из правой части уравнения вычитаются из суммы
				for (int i = 0; i < dim; i++)
				{
					if (elem.value == vars[i]) // Сравнение переменной перед коэффициентом и его позиции в матрице
						koeffs[eq_count][i] -= stod(prev.value);
				}
			}
			if (side && elem.type != TVAR && prev.type == TNUM)
				free_terms[eq_count] += stod(prev.value); // Свободные члены из правой части уравнения вычитаются из суммы
			if (elem.value == "=")
				side = true; // После знака равно начинает рассматриваться правая часть уравнения
			if (elem.type == TEND)
			{
				side = false; // После конца строки начинает рассматриваться левая часть уравнения
				eq_count++;	  // Увеличение счетчика уравнений
			}
			prev = elem; // Копирование текущей просматриваемой лексимы в предыдущую
		}
	}
};

#endif
