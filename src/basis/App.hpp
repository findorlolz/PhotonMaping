#pragma once

#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"
#include "3d/CameraControls.hpp"

class Renderer;
class AssetManager;

namespace FW
{

class App : public Window::Listener
{

private:
    enum Action
    {
        Action_None,
		Action_StartPM,
		Action_ShowPMResult
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
	CameraControls	m_cameraCtrl;
	Action m_action;

	Renderer* m_renderer;
	AssetManager* m_assetManager;

};

}