#include "App.hpp"
#include "base/Main.hpp"
#include "Renderer.h"
#include "AssetManager.h"

using namespace FW;

App::App( void ) : 
	m_commonCtrl( CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5 ),
	m_cameraCtrl(&m_commonCtrl, CameraControls::Feature_All),
	m_action( Action_None )
{
	m_commonCtrl.showFPS(true);
	m_commonCtrl.addButton((S32*)&m_action, Action_StartPM, FW_KEY_1, "Start photon maping...");
	m_commonCtrl.addButton((S32*)&m_action, Action_ShowPMResult, FW_KEY_2, "Render with photon maping...");

	m_window.setTitle("Application");
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);
	m_window.addListener(&m_cameraCtrl);
	m_window.getGL()->swapBuffers();

	m_assetManager = new AssetManager();
	m_assetManager->LoadAssets();

	m_renderer = &Renderer::get();
	m_renderer->startUp(m_window.getGL(), &m_cameraCtrl, m_assetManager);
}

bool App::handleEvent( const Window::Event& event )
{
	if (event.type == Window::EventType_Close)
	{
		m_window.showModalMessage("Exiting...");
		
		m_assetManager->ReleaseAssets();
		delete m_assetManager;
		
		m_renderer->shutDown();

		delete this;
		return true;
	}

	Action action = m_action;
	m_action = Action_None;

	switch (action)
	{
	case Action_None:
		break;

	case Action_StartPM:
		Renderer::get().initPhotonMaping(10000u, m_window.getSize());
		break;

	case Action_ShowPMResult:
		Renderer::get().toggleRenderingMode();
		break;

	default:
		FW_ASSERT(false);
		break;
	}

	m_window.setVisible(true);
	if (event.type == Window::EventType_Paint)
	{
		m_renderer->drawFrame();
	}
	m_window.repaint();

	return false;
}

void FW::init(void) 
{
    new App;
}