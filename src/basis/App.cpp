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
	m_totalLight(10.f),
	m_numberOfPhotons(2000),
	m_numberOfFRRays(1), 
	m_numberOfSamplesPerDimension(1),
	m_visibleCameraControls(false),
	m_stateChange(false),
	m_visiblPMControls(true)
{
	m_commonCtrl.showFPS(true);
	
	m_commonCtrl.beginSliderStack();
	m_commonCtrl.addSlider(&m_FGRadius, 0.01f, 1.0f, true, FW_KEY_NONE, FW_KEY_NONE, "Gathering radius = %f");
	m_commonCtrl.addSlider(&m_numberOfSamplesPerDimension, 1u, 8, false, FW_KEY_NONE, FW_KEY_NONE, "Number of samples per dimension = %d ");
	m_commonCtrl.addSlider(&m_numberOfFRRays, 1, 100, true, FW_KEY_NONE, FW_KEY_NONE, "Number of FG Rays, if zero => no FG = %d ");
	m_commonCtrl.endSliderStack();
	m_commonCtrl.beginSliderStack();
	m_commonCtrl.addSlider(&m_totalLight, .1f, 100.f, true, FW_KEY_NONE, FW_KEY_NONE, "Total light = %f ");
	m_commonCtrl.addSlider(&m_numberOfPhotons, 1000, 500000, true, FW_KEY_NONE, FW_KEY_NONE, "Number of photons = %d");
	m_commonCtrl.endSliderStack();
	
	m_commonCtrl.addButton((S32*)&m_action, Action_StartPM, FW_KEY_1, "Start photon maping...");
	m_commonCtrl.addButton((S32*)&m_action, Action_newImage, FW_KEY_2, "Create image based  on existing PM...");
	m_commonCtrl.addButton((S32*)&m_action, Action_ShowPMResult, FW_KEY_3, "Toggle rendring state...");
	m_commonCtrl.addButton((S32*)&m_action, Action_ControlToggle, FW_KEY_4, "Toggle visibility of PM controls");
	m_commonCtrl.addButton((S32*)&m_action, Action_EnableCamera, FW_KEY_F1, "Enable/Disable camera movement");
	m_commonCtrl.addButton((S32*)&m_action, Action_ChangeCameraState, FW_KEY_F2, "Enable/Disable camera options");

	m_window.setTitle("Application");
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);
	m_window.addListener(&m_cameraCtrl);
	m_window.getGL()->swapBuffers();

	m_cameraCtrl.removeGUIControls();

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
		Renderer::get().initPhotonMaping(m_numberOfPhotons, m_FGRadius, m_numberOfFRRays, m_totalLight, m_numberOfSamplesPerDimension, m_window.getSize());
		m_visiblPMControls = false;
		m_stateChange = true;
		break;

	case Action_newImage:
		Renderer::get().initImageSynthesisFromExistingPM( m_FGRadius, m_numberOfFRRays, m_totalLight, m_numberOfSamplesPerDimension, m_window.getSize());
		m_visiblPMControls = false;
		m_stateChange = true;		
		break;

	case Action_ShowPMResult:
		Renderer::get().toggleRenderingMode();
		m_visiblPMControls = !m_visiblPMControls;
		m_stateChange = true;
		break;

	case Action_ControlToggle:
		m_visiblPMControls = !m_visiblPMControls;
		m_stateChange = true;
		break;

	case Action_ChangeCameraState:
		m_visibleCameraControls = !m_visibleCameraControls;
		m_stateChange = true;
		break;

	case Action_EnableCamera:
		m_commonCtrl.message("Enable/Disable camera movements");
		m_cameraCtrl.setEnableMovement(!m_cameraCtrl.getEnableMovement());
		break;

	default:
		FW_ASSERT(false);
		break;
	}

	if(m_stateChange)
		updateAppState();

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

void App::updateAppState()
{
	if(m_visibleCameraControls)
		m_cameraCtrl.addGUIControls();
	else
		m_cameraCtrl.removeGUIControls();

	m_commonCtrl.showControls(m_visiblPMControls);
	m_stateChange = false;
}