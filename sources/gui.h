#ifndef GUI
#define GUI

#include "slau.h"

class RoundedShape : public ConvexShape
{
private:
	Vector2f size;
public:
	void setSize(const Vector2f& size)
	{
		setPointCount(4);
		setPoint(0, Vector2f(0, 0));
		setPoint(1, Vector2f(size.x, 0));
		setPoint(2, Vector2f(size.x, size.y));
		setPoint(3, Vector2f(0, size.y));
		this->size = size;
	}

	void setRoundRadius(const uint r, const uint point_count = 4)
	{
		setPointCount(0);
		setPointCount(4 * point_count);

		for (int i = 0; i < point_count; i++)
		{
			float x = -static_cast<float>(r) * cos(static_cast<float>(point_count - i) / static_cast<float>(point_count));
			float y = -sqrt(r * r - x * x);
			setPoint(point_count - i - 1, Vector2f(x, y));
		}

		for (int i = 0; i < point_count; i++)
		{
			float x = static_cast<float>(r) * cos(static_cast<float>(point_count - i) / static_cast<float>(point_count));
			float y = -sqrt(r * r - x * x);
			setPoint(point_count + i, Vector2f(size.x + x, y));
		}

		for (int i = 0; i < point_count; i++)
		{
			float x = static_cast<float>(r) * cos(static_cast<float>(point_count - i) / static_cast<float>(point_count));
			float y = sqrt(r * r - x * x);
			setPoint(3 * point_count - i - 1, Vector2f(size.x + x, size.y + y));
		}

		for (int i = 0; i < point_count; i++)
		{
			float x = -static_cast<float>(r) * cos(static_cast<float>(point_count - i) / static_cast<float>(point_count));
			float y = sqrt(r * r - x * x);
			setPoint(3 * point_count + i, Vector2f(x, size.y + y));
		}
		
	}
};

class slau_program
{
private:
	Mat img, current, rs;

	slau_program() {}

	void calcImages()
	{
		resize(img, img, Size(IMG_WIDTH, IMG_HEIGHT));
		int color = calcAvgBnss(img);
		rectangle(img, cv::Rect(Point(0, 0), Point(IMG_WIDTH, 10)), Scalar(color, color, color), 10);
		rectangle(img, cv::Rect(Point(0, 0), Point(10, IMG_HEIGHT)), Scalar(color, color, color), 10);
		rectangle(img, cv::Rect(Point(IMG_WIDTH - 10, 0), Point(IMG_WIDTH, IMG_HEIGHT)), Scalar(color, color, color), 10);
		rectangle(img, cv::Rect(Point(0, IMG_HEIGHT - 10), Point(IMG_WIDTH, IMG_HEIGHT)), Scalar(color, color, color), 10);
		cvtColor(img, current, COLOR_BGR2GRAY);

		vector<Mat> split = splitting(current, IMG_WIDTH, IMG_HEIGHT, 40, 40);
		for (int i = 0; i < split.size(); i++)
			split[i] = preproccesing(split[i], color, [](uint x, int color) {
			double b = 0.05 * color - 10;
			return static_cast<uint>(0.6 * x + b);
				});
		current = unsplitting(split, IMG_WIDTH, IMG_HEIGHT, 40, 40);
		rs = noiseClear(current, NOISE_AREA);

		vector<vector<Point>> ctrs;
		vector<Vec4i> rar;

		findContours(rs, ctrs, rar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		unionNearCtrs(ctrs, [](Point x) {
			return sqrt(x.x * x.x + x.y * x.y) < 15;
			}, [](double a, double b) {
				return a > 2.0 * b;
			});

		detectDraw(img, ctrs);
	}
public:
	string result_str;

	static slau_program* getInstance()
	{
		static slau_program* instance = new slau_program;
		return instance;
	}

	void webcamShow(RenderWindow* w)
	{
		VideoCapture cam(0);
		Mat temp;

		while (true)
		{
			
			string name = "Сканируем СЛАУ";
			cam.read(temp);
			imshow(name, temp);
			waitKey(1);
			Event event;
			bool be_event = w->pollEvent(event);
			if (be_event && event.type == Event::KeyPressed && event.key.code == Keyboard::Enter)
			{
				destroyWindow(name);
				img = temp;
				calcImages();
				break;
			}
		}
	}

	void loadImage(RenderWindow* w)
	{
		ShellExecuteA(NULL, "open", "D:/Работа/Школа Программирования 2023/SLAU/PROJECT/exclude", NULL, NULL, SW_RESTORE);
		while (true)
		{
			Event event;
			bool be_event = w->pollEvent(event);
			if (be_event && event.type == Event::KeyPressed && event.key.code == Keyboard::Enter)
			{
				img = imread("exclude/example.jpg");
				calcImages();
				break;
			}
		}
	}

	string getStr()
	{
		img = imread("exclude/example.jpg");

		vector<Mat> syms;
		getSyms(img, syms);

		vector<uint> layers = { 750, 68, 68, 34 };
		perceptron net(layers);
		net.fscanWeights("data/weights/04.net");

		result_str = "";
		for (int i = 0; i < syms.size(); i++)
		{
			matrixd mat_in = matrixd::cvtCVMat2NNMat(syms[i]);
			vector<double> input = perceptron::cvtImgMat2Vec(mat_in);
			vector<double> output = net.forwardPropagetion(input);
			uint answer = perceptron::softMax(output);
			if (isSeparator(syms[i]))
				result_str += "\n";
			else
				result_str += perceptron::cvtAnswer(answer);
		}

		lexer slau;
		slau.str = result_str;
		slau.tokenizing();
		slau.correction();
		result_str = slau.tok2Str();

		return result_str;
	}

	string solution()
	{
		lexer slau;
		slau.str = result_str;

		matrixd koeffs;
		vector<double> free_terms;
		slau.tokenizing();
		slau.correction();
		slau.exclude(koeffs, free_terms);

		string rs = "";

		try
		{
			vector<string> vars = slau.getVars();
			kramer k(vars.size());
			k.fillMatrix(koeffs);
			k.fillFreeTerms(free_terms);
			vector<double> roots = k.getRoots();

			rs += "DECISION:\n";
			for (int i = 0; i < vars.size(); i++)
				rs += vars[i] + " = " + to_string(roots[i]) + "\t";
			rs += "\n";
		}
		catch (int e)
		{
			if (e == INFINITY_SET_OF_SOLUTIONS)
				rs += "INFINITY SET OF SOLUTIONS\n";
			if (e == OUT_CONVERGENCE)
				rs += "OUT CONVERGENCE\n";
		}

		return rs;
	}

	void showImages()
	{
		imshow("rs", rs);
		imshow("current", current);
		imshow("img", img);
		waitKey(0);
		destroyWindow("rs");
		destroyWindow("current");
		destroyWindow("img");
	}
};

class button
{
private:
	vector<Shape*> units;
	Text name;
	vector<Vector2f> poss;
	Vector2i x, y;
public:
	static vector<Shape*> fillPhotoButton()
	{
		vector<Shape*> units;
		units.reserve(5);
		RoundedShape* shadow = new RoundedShape;
		shadow->setSize(Vector2f(140, 120));
		shadow->setRoundRadius(12, 8);
		shadow->setFillColor(Color(28, 28, 28));
		shadow->setPosition(195, 315);
		units.push_back(shadow);
		RoundedShape* bg = new RoundedShape;
		bg->setSize(Vector2f(140, 120));
		bg->setRoundRadius(12, 8);
		bg->setFillColor(Color(78, 78, 78));
		bg->setPosition(180, 300);
		units.push_back(bg);
		RoundedShape* eye = new RoundedShape;
		eye->setSize(Vector2f(20, 12));
		eye->setRoundRadius(4, 4);
		eye->setOutlineThickness(5);
		eye->setOutlineColor(Color(255, 255, 255));
		eye->setFillColor(Color(78, 78, 78));
		eye->setPosition(250, 315);
		units.push_back(eye);
		RoundedShape* housing = new RoundedShape;
		housing->setSize(Vector2f(100, 60));
		housing->setRoundRadius(8, 4);
		housing->setOutlineThickness(5);
		housing->setOutlineColor(Color(255, 255, 255));
		housing->setFillColor(Color(78, 78, 78));
		housing->setPosition(200, 330);
		units.push_back(housing);
		CircleShape* obj = new CircleShape;
		obj->setRadius(20);
		obj->setOutlineThickness(5);
		obj->setOutlineColor(Color(255, 255, 255));
		obj->setFillColor(Color(78, 78, 78));
		obj->setPosition(240, 350);
		units.push_back(obj);
		return units;
	}

	static vector<Shape*> fillMenuButton()
	{
		vector<Shape*> units;
		units.reserve(4);
		RectangleShape* bg = new RectangleShape;
		bg->setSize(Vector2f(80, 80));
		bg->setFillColor(Color(78, 78, 78));
		units.push_back(bg);
		RoundedShape* line_1 = new RoundedShape;
		line_1->setSize(Vector2f(40, 0));
		line_1->setRoundRadius(4, 4);
		line_1->setFillColor(Color(255, 255, 255));
		line_1->setPosition(20, 20);
		units.push_back(line_1);
		RoundedShape* line_2 = new RoundedShape;
		line_2->setSize(Vector2f(40, 0));
		line_2->setRoundRadius(4, 4);
		line_2->setFillColor(Color(255, 255, 255));
		line_2->setPosition(20, 40);
		units.push_back(line_2);
		RoundedShape* line_3 = new RoundedShape;
		line_3->setSize(Vector2f(40, 0));
		line_3->setRoundRadius(4, 4);
		line_3->setFillColor(Color(255, 255, 255));
		line_3->setPosition(20, 60);
		units.push_back(line_3);
		return units;
	}

	static vector<Shape*> fillBackButton()
	{
		vector<Shape*> units;
		RoundedShape* bg = new RoundedShape;
		bg->setSize(Vector2f(300, 40));
		bg->setRoundRadius(4, 4);
		bg->setFillColor(Color(103, 103, 103));
		bg->setPosition(30, 30);
		units.push_back(bg);
		return units;
	}

	static vector<Shape*> fillInfoButton()
	{
		vector<Shape*> units;
		RoundedShape* bg = new RoundedShape;
		bg->setSize(Vector2f(300, 40));
		bg->setRoundRadius(4, 4);
		bg->setFillColor(Color(103, 103, 103));
		bg->setPosition(30, 80);
		units.push_back(bg);
		return units;
	}

	static vector<Shape*> fillLoadImageButton()
	{
		vector<Shape*> units;
		RoundedShape* bg = new RoundedShape;
		bg->setSize(Vector2f(300, 40));
		bg->setRoundRadius(4, 4);
		bg->setFillColor(Color(103, 103, 103));
		bg->setPosition(30, 130);
		units.push_back(bg);
		return units;
	}

	static vector<Shape*> fillCalcButton()
	{
		vector<Shape*> units;
		RoundedShape* bg = new RoundedShape;
		bg->setSize(Vector2f(300, 40));
		bg->setRoundRadius(4, 4);
		bg->setFillColor(Color(103, 103, 103));
		bg->setPosition(30, 180);
		units.push_back(bg);
		return units;
	}

	static vector<Shape*> fillInputButton()
	{
		vector<Shape*> units;
		RoundedShape* bg = new RoundedShape;
		bg->setSize(Vector2f(300, 40));
		bg->setRoundRadius(4, 4);
		bg->setFillColor(Color(128, 204, 255));
		bg->setPosition(100, 600);
		units.push_back(bg);
		return units;
	}

	uint getUnitsSize() const
	{
		return units.size();
	}

	void getHoverPoint(Vector2i& output_x, Vector2i& output_y) const
	{
		output_x = x;
		output_y = y;
	}

	Shape* operator[](uint index) const
	{
		if (index >= units.size())
			throw OUT_OF_RANGE_GUI_UNITS;
		return units[index];
	}

	Vector2f getPosition(uint index) const
	{
		if (index >= units.size())
			throw OUT_OF_RANGE_GUI_UNITS;
		return poss[index];
	}

	void add(Shape* unit)
	{
		units.push_back(unit);
		poss.push_back(unit->getPosition());
	}

	void fillShapes(vector<Shape*>(*fill)())
	{
		units = fill();
		for (int i = 0; i < units.size(); i++)
			poss.push_back(units[i]->getPosition());
	}

	void setName(const wchar_t* text, Font& font, uint font_size, const Color& font_color)
	{
		name.setFont(font);
		name.setCharacterSize(font_size);
		name.setFillColor(font_color);
		name.setPosition(getPosition(0));
		name.setString(text);
	}

	void setNameColor(const Color& font_color)
	{
		name.setFillColor(font_color);
	}

	void setHoverPoint(const Vector2i& x, const Vector2i& y)
	{
		this->x = x;
		this->y = y;
	}

	bool hover(const Vector2i& pos) const
	{
		return pos.x >= x.x && pos.x <= y.x && pos.y >= x.y && pos.y <= y.y;
	}

	void animate(void(*animation)(button& obj))
	{
		animation(*this);
	}

	void draw(RenderWindow* w) const
	{
		for (int i = 0; i < units.size(); i++)
			w->draw(*units[i]);
		w->draw(name);
	}
};

class board
{
private:
	RoundedShape bg;
	vector<Text> content;

	uint row;
	bool show_content;

	board() 
	{
		bg.setSize(Vector2f(440, 560));
		bg.setRoundRadius(10, 4);
		bg.setFillColor(Color(255, 255, 255));
		bg.setPosition(30, 110);
		row = 100;
		show_content = false;
	}
public:
	static board* getInstance()
	{
		static board* instance = new board;
		return instance;
	}

	void placeContent(const wchar_t* text, const Font& font)
	{
		Text content_unit;
		content_unit.setFont(font);
		content_unit.setCharacterSize(30);
		content_unit.setFillColor(Color(0, 0, 0));
		content_unit.setString(text);
		content_unit.setPosition(50, row);
		row += 30;
		content.push_back(content_unit);
	}

	void placeContent(string text, const Font& font)
	{
		Text content_unit;
		content_unit.setFont(font);
		content_unit.setCharacterSize(30);
		content_unit.setFillColor(Color(0, 0, 0));
		content_unit.setString(text);
		content_unit.setPosition(50, row);
		for (int i = 0; i < text.size(); i++)
		{
			if (text[i] == '\n')
				row += 30;
		}
		content.push_back(content_unit);
	}

	void clearContent()
	{
		row = 100;
		content.clear();
	}

	void beginShow()
	{
		show_content = true;
	}

	void endShow()
	{
		show_content = false;
	}

	void draw(RenderWindow* w)
	{
		if (show_content)
		{
			w->draw(bg);
			for (int i = 0; i < content.size(); i++)
				w->draw(content[i]);
		}
	}
};

class text_bar
{
private:
	string text_show, part;
	Text text_render;
	button input_button;

	RectangleShape cursor;

	bool plus;
	uint counter;
	uint separator;

	text_bar(Font& font) 
	{
		text_show = "";
		part = "";
		input_cont = false;
		input_button.fillShapes(button::fillInputButton);
		input_button.setHoverPoint(Vector2i(95, 600), Vector2i(400, 635));
		input_button.setName(L"    РЕШИТЬ СЛАУ", font, 30, Color(0, 0, 0));
		plus = false;
		cursor.setSize(Vector2f(3, 30));
		cursor.setFillColor(Color(0, 0, 0));
		cursor.setPosition(50, 105);
		counter = 0;
		separator = 0;
	}

	uint checkSym(char sym)
	{
		switch (sym)
		{
		case '-':
			return 0;
		case '+':
			return 1;
		case '=':
			return 1;
		case '0':
			return 2;
		case '1':
			return 3;
		case '2':
			return 4;
		case '3':
			return 5;
		case '4':
			return 6;
		case '5':
			return 7;
		case '6':
			return 8;
		case '7':
			return 9;
		case '8':
			return 10;
		case '9':
			return 11;
		default:
			return 12;
		}
	}
public:
	bool input_cont;

	static text_bar* getInstance(Font& font)
	{
		static text_bar* instance = new text_bar(font);
		return instance;
	}

	void setString(string str)
	{
		text_show = str;
	}

	string getString()
	{
		return text_show + part;
	}

	void input(const Vector2i& pos, Font& font, Event* event)
	{
		uint b[33] = {	10, 16, 18, 16, 18, 17, 16, 17, 17, 17, 16, 17, 18, 18, 18, 18,
						18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18
		};

		if (event->type == Event::KeyPressed)
		{
			int end_counter = 0;
			string full = text_show + part;
			for (int i = 0; i < full.length(); i++)
			{
				if (full[i] == '\n')
					end_counter++;
			}

			switch (event->key.code)
			{
			case 56:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[0], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[0], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "-";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::LShift:
				plus = true;
				break;
			case Keyboard::Equal:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[1], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[1], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				if (plus)
					text_show += "+";
				else
					text_show += "=";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num0:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[2], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[2], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "0";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num1:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[3], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[3], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "1";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num2:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[4], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[4], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "2";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num3:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[5], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[5], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "3";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num4:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[6], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[6], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "4";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num5:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[7], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[7], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "5";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num6:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[8], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[8], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "6";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num7:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[9], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[9], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "7";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num8:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[10], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[10], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "8";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Num9:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[11], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[11], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "9";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::A:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[12], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[12], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "a";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::B:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[13], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[13], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "b";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::C:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[14], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[14], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "c";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::D:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[15], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[15], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "d";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::E:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[16], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[16], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "e";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::F:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[17], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[17], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "f";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::G:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[18], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[18], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "g";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::M:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[19], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[19], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "m";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::N:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[20], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[20], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "n";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::P:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[21], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[21], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "p";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Q:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[22], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[22], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "q";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::R:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[23], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[23], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "r";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::S:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[24], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[24], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "s";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::T:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[25], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[25], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "t";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::U:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[26], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[26], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "u";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::V:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[27], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[27], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "v";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::W:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[28], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[28], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "w";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::X:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[29], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[29], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "x";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Y:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[30], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[30], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "y";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Z:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[31], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[31], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "z";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::I:
				cursor.setPosition(cursor.getPosition() + Vector2f(b[32], 0));
				if (counter >= 22)
				{
					cursor.setPosition(Vector2f(50 + b[32], cursor.getPosition().y + 30));
					text_show += "\n";
					counter = 0;
				}
				text_show += "pi";
				plus = false;
				counter++;
				separator++;
				break;
			case Keyboard::Enter:
				cursor.setPosition(Vector2f(50, cursor.getPosition().y + 30));
				text_show += "\n";
				plus = false;
				counter = 0;
				break;
			case Keyboard::Up:



				plus = false;
				break;
			case Keyboard::Down:



				plus = false;
				break;
			case Keyboard::Left:
				


				plus = false;
				break;
			case Keyboard::Right:
				


				plus = false;
				break;
			case Keyboard::Backspace:
				if (text_show[text_show.length() - 1] == '\n')
				{
					uint bias = 0;
					for (int i = text_show.length() - 2; text_show[i] != '\n' && i >= 0; i--)
						bias += b[checkSym(text_show[i])];
					cursor.setPosition(50 + bias, cursor.getPosition().y - 30);
				}
				else
					cursor.setPosition(cursor.getPosition() - Vector2f(b[checkSym(text_show[text_show.length() - 1])], 0));
				text_show.pop_back();
				plus = false;
				counter--;
				separator--;
				break;
			}
		}

		if (input_button.hover(pos))
			input_button.animate([](button& obj) {
			obj[0]->setFillColor(Color(0, 0, 0));
			obj.setNameColor(Color(128, 204, 255));
				});
		else
			input_button.animate([](button& obj) {
			obj[0]->setFillColor(Color(128, 204, 255));
			obj.setNameColor(Color(0, 0, 0));
				});

		text_render.setFont(font);
		text_render.setCharacterSize(30);
		text_render.setFillColor(Color(0, 0, 0));
		text_render.setPosition(50, 100);
		text_render.setString(text_show + part);
	}

	void click(const Vector2i& pos, Font& font)
	{
		if (input_button.hover(pos) && input_cont)
		{
			input_cont = false;
			board* content = board::getInstance();
			slau_program* program = slau_program::getInstance();
			content->placeContent(text_show + part + '\n', font);
			program->result_str = text_show + part;
			text_show = "";
			part = "";
			content->placeContent(program->solution(), font);
		}
	}

	void draw(RenderWindow* w)
	{
		if (input_cont)
		{
			input_button.draw(w);
			w->draw(text_render);
			w->draw(cursor);
		}
	}
};

class nav_bar
{
private:
	RectangleShape background;
	Font font;
	Text header;

	button menu_button;
	button photo;

	bool menu_draw;
	bool photo_draw;
	RectangleShape menu_background;
	RectangleShape menu_shadow;
	button back;
	button info;
	button load_image;
	button calc;

	nav_bar(Font& font) 
	{
		background.setFillColor(Color(78, 78, 78));
		background.setSize(Vector2f(500, 80));
		header.setFont(font);
		header.setCharacterSize(40);
		header.setFillColor(Color(255, 255, 255));
		header.setPosition(100, 15);
		header.setString(L"Калькулятор СЛАУ");
		menu_button.fillShapes(button::fillMenuButton);
		menu_button.setHoverPoint(Vector2i(0, 0), Vector2i(80, 80));
		photo.fillShapes(button::fillPhotoButton);
		photo.setHoverPoint(Vector2i(168, 290), Vector2i(335, 428));
		menu_draw = false;
		menu_background.setSize(Vector2f(360, 720));
		menu_background.setFillColor(Color(103, 103, 103));
		menu_shadow.setSize(Vector2f(365, 720));
		menu_shadow.setFillColor(Color(28, 28, 28));
		back.fillShapes(button::fillBackButton);
		back.setName(L"НАЗАД", font, 30, Color(255, 255, 255));
		back.setHoverPoint(Vector2i(30, 25), Vector2i(330, 65));
		info.fillShapes(button::fillInfoButton);
		info.setName(L"МЕТОДЫ РЕШЕНИЯ", font, 30, Color(255, 255, 255));
		info.setHoverPoint(Vector2i(30, 75), Vector2i(330, 115));
		load_image.fillShapes(button::fillLoadImageButton);
		load_image.setName(L"ЗАГРУЗИТЬ ФОТО", font, 30, Color(255, 255, 255));
		load_image.setHoverPoint(Vector2i(30, 125), Vector2i(330, 165));
		calc.fillShapes(button::fillCalcButton);
		calc.setName(L"КАЛЬКУЛЯТОР", font, 30, Color(255, 255, 255));
		calc.setHoverPoint(Vector2i(30, 175), Vector2i(330, 215));
		photo_draw = true;
	}
public:
	static nav_bar* getInstance(Font& font)
	{
		static nav_bar* instance = new nav_bar(font);
		return instance;
	}

	void render(const Vector2i& pos)
	{
		if (menu_button.hover(pos))
			menu_button.animate([](button& obj) {
			obj[0]->setFillColor(Color(128, 204, 255));
				});
		else
			menu_button.animate([](button& obj) {
			obj[0]->setFillColor(Color(78, 78, 78));
				});
		if (photo.hover(pos))
			photo.animate([](button& obj) {
			for (int i = 1; i < obj.getUnitsSize(); i++)
			{
				obj[i]->setOutlineColor(Color(128, 204, 255));
				Vector2f pos = obj.getPosition(i);
				obj[i]->setPosition(pos + Vector2f(10, 10));
			}
				});
		else
			photo.animate([](button& obj) {
			for (int i = 1; i < obj.getUnitsSize(); i++)
			{
				obj[i]->setOutlineColor(Color(255, 255, 255));
				obj[i]->setPosition(obj.getPosition(i));
			}
				});
		if (back.hover(pos))
			back.animate([](button& obj) {
			obj[0]->setFillColor(Color(255, 255, 255));
			obj.setNameColor(Color(128, 204, 255));
				});
		else
			back.animate([](button& obj) {
			obj[0]->setFillColor(Color(103, 103, 103));
			obj.setNameColor(Color(255, 255, 255));
				});
		if (info.hover(pos))
			info.animate([](button& obj) {
			obj[0]->setFillColor(Color(255, 255, 255));
			obj.setNameColor(Color(128, 204, 255));
				});
		else
			info.animate([](button& obj) {
			obj[0]->setFillColor(Color(103, 103, 103));
			obj.setNameColor(Color(255, 255, 255));
				});
		if (load_image.hover(pos))
			load_image.animate([](button& obj) {
			obj[0]->setFillColor(Color(255, 255, 255));
			obj.setNameColor(Color(128, 204, 255));
				});
		else
			load_image.animate([](button& obj) {
			obj[0]->setFillColor(Color(103, 103, 103));
			obj.setNameColor(Color(255, 255, 255));
				});
		if (calc.hover(pos))
			calc.animate([](button& obj) {
			obj[0]->setFillColor(Color(255, 255, 255));
			obj.setNameColor(Color(128, 204, 255));
				});
		else
			calc.animate([](button& obj) {
			obj[0]->setFillColor(Color(103, 103, 103));
			obj.setNameColor(Color(255, 255, 255));
				});
	}

	void click(const Vector2i& pos, Font& font, RenderWindow* w)
	{
		if (photo.hover(pos) && photo_draw)
		{
			slau_program* program = slau_program::getInstance();
			program->webcamShow(w);
			program->showImages();
			board* content = board::getInstance();
			content->beginShow();
			photo_draw = false;
			text_bar* text = text_bar::getInstance(font);
			text->input_cont = true;
			string str = program->getStr();
			text->setString(str);
		}
		else if (menu_button.hover(pos))
			menu_draw = true;
		else if (back.hover(pos))
		{
			menu_draw = false;
			board* content = board::getInstance();
			content->endShow();
			content->clearContent();
			photo_draw = true;
		}
		else if (info.hover(pos))
		{
			menu_draw = false;
			board* content = board::getInstance();
			content->beginShow();
			photo_draw = false;
			content->placeContent(L"Калькулятор СЛАУ способен", font);
			content->placeContent(L"решать СЛАУ с помощью", font);
			content->placeContent(L"пяти методов на выбор:", font);
			content->placeContent(L"1. Метод Крамера", font);
			content->placeContent(L"2. Метод Гаусса", font);
			content->placeContent(L"3. Метод Жорадна-Гаусса", font);
			content->placeContent(L"4. Метод Простой Итерации", font);
			content->placeContent(L"5. Метод Зейделя", font);
		}
		else if (load_image.hover(pos))
		{
			menu_draw = false;
			slau_program* program = slau_program::getInstance();
			program->loadImage(w);
			program->showImages();
			board* content = board::getInstance();
			content->beginShow();
			photo_draw = false;
			text_bar* text = text_bar::getInstance(font);
			text->input_cont = true;
			string str = program->getStr();
			text->setString(str);
		}
		else if (calc.hover(pos))
		{
			menu_draw = false;
			board* content = board::getInstance();
			content->beginShow();
			photo_draw = false;
			text_bar* text = text_bar::getInstance(font);
			text->input_cont = true;
		}
	}

	void draw(RenderWindow* w)
	{
		w->draw(background);
		w->draw(header);
		menu_button.draw(w);
		if (photo_draw)
			photo.draw(w);
		if (menu_draw)
		{
			w->draw(menu_shadow);
			w->draw(menu_background);
			back.draw(w);
			info.draw(w);
			load_image.draw(w);
			calc.draw(w);
		}
	}
};

#endif