#include "App.hpp"
#include "base/Main.hpp"
#include "Renderer.h"
#include "AssetManager.h"

using namespace FW;

App::App( void ) : 
	m_commonCtrl( CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5 ),
	m_cameraCtrl(&m_commonCtrl, CameraControls::Feature_All),
	m_action( Action_None ),
	m_FGRadius(.2f),
	m_totalLight(100.f),
	m_numberOfPhotons(1000),
	m_numberOfFRRays(1)

{
	m_commonCtrl.showFPS(true);
	
	m_commonCtrl.beginSliderStack();
	m_commonCtrl.addSlider(&m_FGRadius, 0.01f, 1.0f, true, FW_KEY_NONE, FW_KEY_NONE, "Gathering radius = %f");
	m_commonCtrl.addSlider(&m_totalLight, 10.f, 10000.f, true, FW_KEY_NONE, FW_KEY_NONE, "Total light = %f ");
	m_commonCtrl.addSlider(&m_numberOfPhotons, 1, 50000, true, FW_KEY_NONE, FW_KEY_NONE, "Number of photons = %d");
	m_commonCtrl.addSlider(&m_numberOfFRRays, 1, 100, true, FW_KEY_NONE, FW_KEY_NONE, "Number of FG Rays, if zero => no FG = %d ");
	m_commonCtrl.endSliderStack();
	
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
		Renderer::get().initPhotonMaping(m_numberOfPhotons, m_FGRadius, m_numberOfFRRays, m_totalLight, m_window.getSize());
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