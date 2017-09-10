#include "MainWindow.h"

#include <algorithm>
#include <iostream>
#include <thread>

#include <QtWidgets/qgridlayout.h>
#include <QtWidgets/qmessagebox.h>
#include <QtCore/qstandardpaths.h>
#include <QtGui/qguiapplication.h>
#include <QtGui/qimagereader.h>
#include <QtGui/qimagewriter.h>


MainWindow::MainWindow(QWidget* parent) :
	QMainWindow(parent)
{
	// Creating window layout

	int width = 600;
	int height = 350;
	int padding = 9;

	resize(width, height);
	setWindowTitle("npainter");
	
	QWidget* centralwidget = new QWidget(this);

	QGridLayout* gridLayout = new QGridLayout(centralwidget);
	gridLayout->setContentsMargins(0, 0, 0, 0);
	gridLayout->setMargin(padding);

	// Creating images
	m_labelRight = new QLabel(centralwidget);
	m_labelRight->setAlignment(Qt::AlignCenter);
	gridLayout->addWidget(m_labelRight, 0, 1, 1, 1);

	m_labelLeft = new QLabel(centralwidget);
	m_labelLeft->setAlignment(Qt::AlignCenter);
	gridLayout->addWidget(m_labelLeft, 0, 0, 1, 1);

	// Create line
	QFrame* line = new QFrame(centralwidget);
	line->setFrameShape(QFrame::HLine);
	line->setFrameShadow(QFrame::Sunken);
	gridLayout->addWidget(line, 1, 0, 1, 2);

	// Creating buttons
	m_buttonTrainingSource = new QPushButton(centralwidget);
	m_buttonTrainingSource->setText("Select training source");
	gridLayout->addWidget(m_buttonTrainingSource, 2, 0, 1, 1);

	m_buttonTrainingOutput = new QPushButton(centralwidget);
	m_buttonTrainingOutput->setText("Select training output");
	gridLayout->addWidget(m_buttonTrainingOutput, 2, 1, 1, 1);

	m_buttonSource = new QPushButton(centralwidget);
	m_buttonSource->setText("Select input image");
	gridLayout->addWidget(m_buttonSource, 3, 0, 1, 1);

	m_buttonEvaluate = new QPushButton(centralwidget);
	m_buttonEvaluate->setText("Evaluate");
	gridLayout->addWidget(m_buttonEvaluate, 3, 1, 1, 1);

	// Assigning
	setCentralWidget(centralwidget);


	// Creating window events
	connect(m_buttonTrainingSource, &QPushButton::pressed, this, &MainWindow::onSelectTrainingSource);
	connect(m_buttonTrainingOutput, &QPushButton::pressed, this, &MainWindow::onSelectTrainingOutput);
	connect(m_buttonSource, &QPushButton::pressed, this, &MainWindow::onSelectInput);
	connect(m_buttonEvaluate, &QPushButton::pressed, this, &MainWindow::onEvaluate);


	// Initializing neural network
	m_network = fann_create_standard(3, 27, 8, 3);
	fann_set_activation_function_hidden(m_network, FANN_SIGMOID);
	fann_set_activation_function_output(m_network, FANN_SIGMOID);
}

MainWindow::~MainWindow()
{
	fann_destroy(m_network);
}

void MainWindow::initializeImageFileDialog(QFileDialog & dialog, QFileDialog::AcceptMode acceptMode)
{
	static bool firstDialog = true;

	if (firstDialog) {
		firstDialog = false;
		const QStringList picturesLocations = QStandardPaths::standardLocations(QStandardPaths::PicturesLocation);
		if (picturesLocations.isEmpty()) {
			dialog.setDirectory(QDir::currentPath());
		}
		else {
			dialog.setDirectory(picturesLocations.last());
		}
	}

	QStringList mimeTypeFilters;

	QByteArrayList supportedMimeTypes;
	if (acceptMode == QFileDialog::AcceptOpen) {
		supportedMimeTypes = QImageReader::supportedMimeTypes();
	}
	else {
		supportedMimeTypes = QImageWriter::supportedMimeTypes();
	}

	for (const QByteArray& mimeTypeName : supportedMimeTypes) {
		mimeTypeFilters.append(mimeTypeName);
	}

	mimeTypeFilters.sort();
	dialog.setMimeTypeFilters(mimeTypeFilters);
	dialog.selectMimeTypeFilter("image/png");

	if (acceptMode == QFileDialog::AcceptSave) {
		dialog.setDefaultSuffix("png");
	}
}

std::unique_ptr<QImage> MainWindow::loadFile(const QString & fileName)
{
	QImageReader reader(fileName);
	reader.setAutoTransform(true);

	std::unique_ptr<QImage> image = std::make_unique<QImage>(reader.read());

	if (image == nullptr || image->isNull()) {
		QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
			tr("Cannot load %1: %2")
			.arg(QDir::toNativeSeparators(fileName), reader.errorString()));
		return nullptr;
	}

	return std::move(image);
}


// Main events handling //
//////////////////////////

void MainWindow::onSelectTrainingSource()
{
	QFileDialog dialog(this, tr("Open File"));
	initializeImageFileDialog(dialog, QFileDialog::AcceptOpen);

	while (dialog.exec() == QDialog::Accepted && 
		!(m_trainingSource = loadFile(dialog.selectedFiles().first())))
	{
	}

	if (m_trainingSource != nullptr) {
		m_trainingSource->convertToFormat(QImage::Format_RGB32);

		m_labelLeft->setPixmap(QPixmap::fromImage(*m_trainingSource));
	}
}

void MainWindow::onSelectTrainingOutput()
{
	QFileDialog dialog(this, tr("Open File"));
	initializeImageFileDialog(dialog, QFileDialog::AcceptOpen);

	while (dialog.exec() == QDialog::Accepted &&
		!(m_trainingOutput = loadFile(dialog.selectedFiles().first())))
	{
	}

	if (m_trainingOutput != nullptr) {
		m_trainingOutput->convertToFormat(QImage::Format_RGB32);
		m_labelRight->setPixmap(QPixmap::fromImage(*m_trainingOutput));
	}
}

void MainWindow::onSelectInput()
{
	QFileDialog dialog(this, tr("Open File"));
	initializeImageFileDialog(dialog, QFileDialog::AcceptOpen);

	while (dialog.exec() == QDialog::Accepted &&
		!(m_inputImage = loadFile(dialog.selectedFiles().first())))
	{
	}

	if (m_inputImage != nullptr) {
		m_inputImage->convertToFormat(QImage::Format_RGB32);
		m_resultImage = std::make_unique<QImage>(m_inputImage->size(), QImage::Format_RGB32);

		m_labelLeft->setPixmap(QPixmap::fromImage(*m_inputImage));
		m_labelRight->setPixmap(QPixmap::fromImage(*m_resultImage));
	}
}

void MainWindow::onEvaluate()
{
	m_buttonEvaluate->setEnabled(false);

	std::thread([this]() {
		for (size_t i = 0; i < 100; ++i) {
			train();
			preview();
		}

		m_buttonEvaluate->setEnabled(true);
	}).detach();
}


void MainWindow::train()
{
	QSize size = m_trainingSource->size();

	static QPoint neighbours[9] = {
		QPoint(0,  0),
		QPoint(-1, -1),
		QPoint(0, -1),
		QPoint(1, -1),
		QPoint(-1,  0),
		QPoint(1,  0),
		QPoint(-1,  1),
		QPoint(0,  1),
		QPoint(1,  1),
	};

	const QRgb* trainingSource = reinterpret_cast<const QRgb*>(m_trainingSource->bits());
	const QRgb* trainingOutput = reinterpret_cast<const QRgb*>(m_trainingOutput->bits());

	for (int y = 0; y < size.height(); ++y) {
		for (int x = 0; x < size.width(); ++x) {
			QPoint point(x, y);

			std::vector<double> pixels(27);
			for (size_t i = 0; i < 9; ++i) {
				QPoint pointToSelect = point + neighbours[i];

				if (pointToSelect.x() < 0) {
					pointToSelect.setX(0);
				}
				if (pointToSelect.y() < 0) {
					pointToSelect.setY(0);
				}

				if (pointToSelect.x() >= size.width()) {
					pointToSelect.setX(size.width() - 1);
				}
				if (pointToSelect.y() >= size.height()) {
					pointToSelect.setY(size.height() - 1);
				}

				QRgb color = trainingSource[pointToSelect.y() * size.width() + pointToSelect.x()];
				pixels[i * 3 + 0] = static_cast<double>(qRed(color)) / 255.0;
				pixels[i * 3 + 1] = static_cast<double>(qGreen(color)) / 255.0;
				pixels[i * 3 + 2] = static_cast<double>(qBlue(color)) / 255.0;
			}

			QRgb color = trainingOutput[y * size.width() + x];
			std::vector<double> targetColor = { 
				static_cast<double>(qRed(color)) / 255.0,
				static_cast<double>(qGreen(color)) / 255.0,
				static_cast<double>(qBlue(color)) / 255.0
			};
			
			fann_train(m_network, pixels.data(), targetColor.data());
		}
	}
}

void MainWindow::preview()
{
	if (m_inputImage == nullptr || m_resultImage == nullptr) {
		return;
	}

	QSize size = m_inputImage->size();

	static QPoint neighbours[9] = {
		QPoint(0,  0),
		QPoint(-1, -1),
		QPoint(0, -1),
		QPoint(1, -1),
		QPoint(-1,  0),
		QPoint(1,  0),
		QPoint(-1,  1),
		QPoint(0,  1),
		QPoint(1,  1),
	};

	const QRgb* inputImage = reinterpret_cast<const QRgb*>(m_inputImage->bits());
	QRgb* resultImage = reinterpret_cast<QRgb*>(m_resultImage->bits());

	for (int y = 0; y < size.height(); ++y) {
		for (int x = 0; x < size.width(); ++x) {
			QPoint point(x, y);

			std::vector<double> pixels(27);
			for (size_t i = 0; i < 9; ++i) {
				QPoint pointToSelect = point + neighbours[i];

				if (pointToSelect.x() < 0) {
					pointToSelect.setX(0);
				}
				if (pointToSelect.y() < 0) {
					pointToSelect.setY(0);
				}

				if (pointToSelect.x() >= size.width()) {
					pointToSelect.setX(size.width() - 1);
				}
				if (pointToSelect.y() >= size.height()) {
					pointToSelect.setY(size.height() - 1);
				}

				QRgb color = inputImage[pointToSelect.y() * size.width() + pointToSelect.x()];
				pixels[i * 3 + 0] = static_cast<double>(qRed(color)) / 255.0;
				pixels[i * 3 + 1] = static_cast<double>(qGreen(color)) / 255.0;
				pixels[i * 3 + 2] = static_cast<double>(qBlue(color)) / 255.0;
			}

			double* newColor = fann_run(m_network, pixels.data());

			resultImage[y * size.width() + x] = qRgb(newColor[0] * 255.0, newColor[1] * 255, newColor[2] * 255);
		}
	}

	m_labelRight->setPixmap(QPixmap::fromImage(*m_resultImage));
}
