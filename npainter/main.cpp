#include <QtWidgets/qapplication.h>
#include <QtWidgets/qmessagebox.h>

#include "MainWindow.h"

int main(int argc, char** argv)
{
	QApplication::addLibraryPath("./");

	QApplication app(argc, argv);

	MainWindow mainWindow;
	mainWindow.show();

	try {
		int result = app.exec();
		return result;
	}
	catch (const std::exception& e) {
		QMessageBox::warning(nullptr, "Exception", e.what());
	}
}