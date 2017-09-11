#pragma once

#include <memory>
#include <mutex>

#include <QtWidgets/qmainwindow.h>
#include <QtWidgets/qpushbutton.h>
#include <QtWidgets/qfiledialog.h>
#include <QtWidgets/qlabel.h>

#include <doublefann.h>

class MainWindow : public QMainWindow
{
public:
	MainWindow(QWidget* parent = nullptr);
	~MainWindow();

private:
	void onSelectTrainingSource();
	void onSelectTrainingOutput();
	void onSelectInput();
	void onEvaluate();

	void train();
	void preview();

	void initializeImageFileDialog(QFileDialog& dialog, QFileDialog::AcceptMode acceptMode);
	std::unique_ptr<QImage> loadFile(const QString& fileName);
	std::vector<QPoint> generateKernel(size_t size);

	QLabel* m_labelLeft;
	QLabel* m_labelRight;

	QPushButton* m_buttonTrainingSource;
	QPushButton* m_buttonTrainingOutput;
	QPushButton* m_buttonSource;
	QPushButton* m_buttonEvaluate;

	std::unique_ptr<QImage> m_trainingSource;
	std::unique_ptr<QImage> m_trainingOutput;

	std::unique_ptr<QImage> m_inputImage;
	std::unique_ptr<QImage> m_resultImage;

	fann* m_network;
	std::vector<QPoint> m_kernel;

	std::mutex m_evaluationMutex;

	bool m_isEvaluating;
};