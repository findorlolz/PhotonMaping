#include "App.hpp"
#include "base/Main.hpp"

using namespace FW;

App::App( void ) : 
	m_commonCtrl( CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5 ),
	m_action( Action_None )
{
	m_commonCtrl.showFPS(true);
	m_commonCtrl.addButton((S32*)&m_action, Action_Temp, FW_KEY_NONE, "Temp action");

	m_window.setTitle("Application");
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);

	m_window.getGL()->swapBuffers();
}

bool App::handleEvent( const Window::Event& event )
{
	if (event.type == Window::EventType_Close)
	{
		m_window.showModalMessage("Exiting...");
		delete this;
		return true;
	}

	Action action = m_action;
	m_action = Action_None;

	switch (action)
	{
	case Action_None:
		break;

	case Action_Temp:
		m_commonCtrl.message("Temp action");
		break;

	default:
		FW_ASSERT(false);
		break;
	}

	m_window.setVisible(true);
	if (event.type == Window::EventType_Paint)
	{
		// Draw something
	}
	m_window.repaint();

	return false;
}

void FW::init(void) 
{
    new App;
}