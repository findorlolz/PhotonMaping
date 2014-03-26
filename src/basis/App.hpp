#pragma once

#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"

namespace FW
{

class App : public Window::Listener
{

private:
    enum Action
    {
        Action_None,
		Action_Temp
    };

public:
	App( void );
	virtual ~App( void ) {}
	virtual bool handleEvent( const Window::Event& event );

private:
	App( const App& ); // forbidden
    App& operator=( const App& ); // forbidden

private:
	Window m_window;
	CommonControls m_commonCtrl;
	Action m_action;

};

}