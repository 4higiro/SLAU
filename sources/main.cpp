#include "slau.h"

#define CONSOLE_VERSION

void main()
{
#ifdef GUI_VERSION
	setlocale(LC_ALL, "Ru");

	Vector2f resolution = Vector2f(500, 720);

	RenderWindow window(VideoMode(resolution.x, resolution.y), L"SLAU");
	VideoMode vm = VideoMode::getDesktopMode();
	window.setFramerateLimit(60);

	Font font;
	font.loadFromFile("resources/gg_kizhich.otf");

	Font font_bold;
	font_bold.loadFromFile("resources/gg_kizhich_bold.otf");

	nav_bar* nav = nav_bar::getInstance(font_bold);
	board* content = board::getInstance();
	slau_program* program = slau_program::getInstance();
	text_bar* text = text_bar::getInstance(font);

	Text message;
	message.setFont(font);
	message.setCharacterSize(20);
	message.setFillColor(Color(255, 255, 255));
	message.setPosition(90, 480);
	message.setString(L"Нажмите, чтобы решить СЛАУ");

	while (window.isOpen())
	{
		Event event;
		while (window.pollEvent(event))
		{
			Vector2i mouse_pos = Mouse::getPosition(window);
			nav->render(mouse_pos);
			text->input(mouse_pos, font, &event);
			switch (event.type)
			{
			case Event::Closed:
				window.close();
				break;
			case Event::MouseButtonPressed:
				switch (event.mouseButton.button)
				{
				case Mouse::Left:
					nav->click(mouse_pos, font, &window);
					text->click(mouse_pos, font);
					break;
				}
				break;
			case Event::KeyPressed:
				switch (event.key.code)
				{
				case Keyboard::Escape:
					window.close();
					break;
				}
				break;
			}
		}
		window.draw(message);
		content->draw(&window);
		text->draw(&window);
		nav->draw(&window);
		window.display();
		window.clear(Color(56, 56, 56));
	}

#endif 

#ifdef CONSOLE_VERSION
	Mat current, rs, img = imread("data/examples/example_14.jpg");
	resize(img, img, Size(IMG_WIDTH, IMG_HEIGHT));
	int color = calcAvgBnss(img);
	rectangle(img, cv::Rect(Point(0, 0), Point(IMG_WIDTH, 10)), Scalar(color, color, color), 10);
	rectangle(img, cv::Rect(Point(0, 0), Point(10, IMG_HEIGHT)), Scalar(color, color, color), 10);
	rectangle(img, cv::Rect(Point(IMG_WIDTH - 10, 0), Point(IMG_WIDTH, IMG_HEIGHT)), Scalar(color, color, color), 10);
	rectangle(img, cv::Rect(Point(0, IMG_HEIGHT - 10), Point(IMG_WIDTH, IMG_HEIGHT)), Scalar(color, color, color), 10);
	cvtColor(img, current, COLOR_BGR2GRAY);
	
	vector<Mat> syms;
	getSyms(img, syms);
	
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

	vector<uint> layers = { 750, 68, 68, 34 };
	perceptron net(layers);
	net.fscanWeights("data/weights/04.net");

	string result_str = "";
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

	cout << "DETECTED:" << endl << result_str << endl;
	lexer slau;
	slau.str = result_str;
	slau.tokenizing();
	slau.correction();
	result_str = slau.tok2Str();
	cout << endl << "AFTER CORRECTION: "<< endl << result_str << endl;

	matrixd koeffs;
	vector<double> free_terms;
	slau.exclude(koeffs, free_terms);

	cout << "MATRIX:" << endl;
	for (int i = 0; i < koeffs.rCount(); i++)
	{
		for (int j = 0; j < koeffs.cCount(); j++)
			cout << koeffs[i][j] << "\t";
		cout << endl;
	}
	cout << "FREE TERMS:" << endl;
	for (int i = 0; i < free_terms.size(); i++)
		cout << free_terms[i] << "\t";
	cout << endl << endl;

	try
	{
		vector<string> vars = slau.getVars();
		kramer k(vars.size());
		k.fillMatrix(koeffs);
		k.fillFreeTerms(free_terms);
		vector<double> roots = k.getRoots();

		cout << "DECISION:" << endl;
		for (int i = 0; i < vars.size(); i++)
			cout << vars[i] << " = " << roots[i] << "\t";
		cout << endl;
	}
	catch (int e)
	{
		if (e == INFINITY_SET_OF_SOLUTIONS)
			cout << "INFINITY SET OF SOLUTIONS" << endl;
		if (e == OUT_CONVERGENCE)
			cout << "OUT CONVERGENCE" << endl;
	}
	
	imshow("rs", rs);
	imshow("current", current);
	imshow("img", img);
	waitKey(0);

#endif

#ifdef LEARNING
	vector<example> dataset;
	vector<example> test;

	cout << "Loading dataset..." << endl;

	ifstream fin;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < SYM_COUNT; j++)
		{
			string path = "data/dataset/dist_" + to_string(i + 1) + "/data_" + perceptron::cvtAnswer(j) + ".txt";
			fin.open(path);
			while (!fin.eof())
			{
				matrixd matrix;
				matrix.resize(MAT_HEIGHT, MAT_WIDTH);
				for (int k = 0; k < MAT_HEIGHT; k++)
				{
					for (int l = 0; l < MAT_WIDTH; l++)
						fin >> matrix[k][l];
				}
				example exmp = { matrix, j };
				dataset.push_back(exmp);
			}
			fin.close();
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < SYM_COUNT; j++)
		{
			string path = "data/dataset/dist_" + to_string(i + 1) + "/data_" + perceptron::cvtAnswer(j) + ".txt";
			fin.open(path);
			matrixd matrix;
			matrix.resize(MAT_HEIGHT, MAT_WIDTH);
			for (int k = 0; k < MAT_HEIGHT; k++)
			{
				for (int l = 0; l < MAT_WIDTH; l++)
					fin >> matrix[k][l];
			}
			example exmp = { matrix, j };
			test.push_back(exmp);
			fin.close();
		}
	}

	system("cls");

	random_shuffle(begin(dataset), end(dataset));

	vector<uint> layers = { 750, 68, 68, 34 };
	perceptron net(layers);
	net.learning_rate = 0.2;	// 0.2
	net.moment = 0.07;			// 0.07

	cout << "Begin learning..." << endl;

	for (int i = 0; i < dataset.size(); i++)
	{
		cout << "Learning progress: " << i + 1 << " / " << dataset.size() << endl;
		vector<double> input = perceptron::cvtImgMat2Vec(dataset[i].img);
		vector<double> pred = perceptron::cvtAnswer2Vec(dataset[i].answer);
		differential diff = net.backPropagetion(input, pred);
		net.learn(diff);
		system("cls");
	}

	net.fprintWeights("data/04.net");

	for (int i = 0; i < test.size(); i++)
	{
		vector<double> input = perceptron::cvtImgMat2Vec(test[i].img);
		vector<double> output = net.forwardPropagetion(input);
		uint answer = perceptron::softMax(output);
		cout << perceptron::cvtAnswer(test[i].answer) << "(" << perceptron::cvtAnswer(answer) << ")" << endl;
		for (int j = 0; j < output.size(); j++)
			cout << output[j] << "\t";
		cout << endl << endl;
	}

#endif

#ifdef CREATE_DATASET
	string path = "data/dataset/src_3/data_" + (string)CURRENT_SYM + ".jpg";
	Mat img = imread(path);
	Mat current, rs;

#ifdef SEE_RESULT

	resize(img, img, Size(750, 1100));
#endif

#ifdef TUNING

	resize(img, img, Size(), 0.25f, 0.25f);
#endif

	cvtColor(img, current, COLOR_BGR2HSV);

	int hmin = 0, hmax = 179;
	int smin = 0, smax = 255;
	int vmin = 0, vmax = VALUE_MAX;

#ifdef SEE_RESULT

	inRange(current, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), current);
	Mat ker = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	dilate(current, current, ker);
#endif

#ifdef TUNING

	namedWindow("tracks");
	createTrackbar("vmax", "tracks", &vmax, 255);

	Mat temp = current;

	while (true)
	{
		inRange(temp, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), current);
		imshow("test", current);
		waitKey(1);
	}
#endif

#ifdef SEE_RESULT

	vector<vector<Point>> ctrs;
	vector<Vec4i> rar;

	rs = noiseClear(current, NOISE_AREA - 24);
	findContours(rs, ctrs, rar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	unionNearCtrs(ctrs, [](Point x) {
		return sqrt(x.x * x.x + x.y * x.y) < 10;
		});

	vector<Mat> dataset;
	fragmentation(current, ctrs, dataset, 4, 4);

	cout << dataset.size() << endl;

	for (int i = 0; i < dataset.size(); i++)
		resize(dataset[i], dataset[i], Size(MAT_WIDTH, MAT_HEIGHT));
#endif

#ifdef OUTPUT_DATASET

	path = "data/data_" + (string)CURRENT_SYM + ".txt";
	ofstream fout;
	fout.open(path);

	for (int k = 0; k < dataset.size(); k++)
	{
		matrixd matrix = matrixd::cvtCVMat2NNMat(dataset[k]);
		for (int i = 0; i < MAT_HEIGHT; i++)
		{
			for (int j = 0; j < MAT_WIDTH; j++)
				fout << matrix[i][j]; << "  ";
			fout << endl;
		}
		fout << endl << endl << endl;
	}

	fout.close();
#endif

#ifdef SEE_RESULT

	detectDraw(img, ctrs);
	resize(img, img, Size(), 0.5f, 0.5f);
	resize(current, current, Size(), 0.5f, 0.5f);
	resize(rs, rs, Size(), 0.5f, 0.5f);

	imshow("img", img);
	imshow("current", current);
	imshow("rs", rs);
	waitKey(0);
#endif

#endif
}