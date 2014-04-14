#pragma once

#include "AssetManager.h"
#include "3d/CameraControls.hpp"
#include "HelpFunctions.h"
#include "base/Random.hpp"
#include "base/MulticoreLauncher.hpp"

typedef FW::Mesh<FW::VertexPNTC> MeshC;

class RayTracer;
class Sampling;

enum MaterialPM
{
	MaterialPM_Lightsource,
	MaterialPM_Diffuse,
	MaterialPM_Mirror,
	MaterialPM_None
};

class Renderer
{
public:

        static Renderer& get()
        {
			static Renderer* gpSingleton = nullptr;
			if (gpSingleton == nullptr)
			{
					gpSingleton = new Renderer();
			}
			FW_ASSERT(gpSingleton != nullptr && "Failed to create Renderer");
			return *gpSingleton;
        }

		void startUp(FW::GLContext*, FW::CameraControls*, AssetManager*);
        void shutDown();

		void drawFrame();
		void initPhotonMaping(const size_t, const float, const size_t, const float, const size_t, const FW::Vec2i&);
		void initImageSynthesisFromExistingPM(const float, const size_t, const float, const size_t, const FW::Vec2i&);
		void clearTriangles();
		void toggleRenderingMode() 
		{ 
			if(!m_hasPhotonMap)
			{
				std::cout << "No image synthesis done yet, just basic rendering available!" << std::endl;
				std::cout << "Press 1 for photong casting and image synthesis" << std::endl;
				std::cout << "___________________________________________________________________" << std::endl;
				return;
			}
			m_renderWithPhotonMaping = !m_renderWithPhotonMaping; 
		}

private:

		 Renderer() {}
        ~Renderer() {}

		void castIndirectLight(const Photon&, const Hit& hit, Node**, std::vector<Hit>&);
		void castDirectLight(const size_t, std::vector<Hit>&);
		void preCalculateOutgoingLight(std::vector<Hit>&);
		static void outgoingLightFunc(FW::MulticoreLauncher::Task&);

		void initTrianglesFromMesh(MeshType, const FW::Vec3f&);
		void updateTriangleToMeshDataPointers();

		void drawTriangleToCamera(const FW::Vec3f& pos, const FW::Vec4f& color);
		void drawPhotonMap();

		void updateContext(const float, const size_t, const float, const size_t);
		void updatePhotonListCapasity(const size_t);

		void synthesisImage(const FW::Vec2i& size);
		static void imageScanline(FW::MulticoreLauncher::Task&);

		struct contextData
		{
			FW::Mat4f d_invP;
			size_t d_numberOfFGRays;
			size_t d_numberOfSamplesByDimension;
			float d_FGRadius;
			float d_totalLight;
			std::vector<FW::Vec3f>* d_vertices;
			std::vector<Triangle>* d_triangles;
			std::vector<TriangleToMeshData>* d_triangleToMeshData;
			std::vector<size_t>* d_indexListFromScene;
			std::vector<Photon>* d_photons;
			std::vector<size_t>* d_photonIndexList;
			std::vector<Hit>* d_tmpHitList;
			FW::Image* d_image;
			MeshC* d_mesh;
			Node* d_sceneTree;
			Node* d_photonTree;		
		};

		static FW::Vec4f interpolateAttribute(const Triangle& tri, const FW::Vec3f&, const FW::MeshBase* mesh, int attribidx);
		static FW::Vec3f getDiversion(const FW::Vec3f&, const Triangle&);
		static FW::Vec3f getAlbedo(const TriangleToMeshData*, const MeshC*, const FW::Vec3f&);
		static FW::Vec3f randomVectorToHalfUnitSphere(const FW::Vec3f&, FW::Random&);
		static MaterialPM traceRay(const FW::Vec3f&, const FW::Vec3f&, Hit&, const std::vector<Triangle>&, const std::vector<size_t>&, Node*, Node**, MeshC*);
		
		static FW::Vec3f finalGathering(const Hit&, const FW::Vec3f&, const contextData&, Node**);

		static MaterialPM shader(const Hit&, MeshC*);

		FW::Random m_randomGen;
		
		std::vector<FW::Vec3f> m_vertices;
		std::vector<Triangle> m_triangles;
		std::vector<size_t> m_lightSources;
		std::vector<TriangleToMeshData> m_triangleToMeshData;
		std::vector<size_t> m_indexListFromScene;
		std::vector<Photon> m_photons;
		std::vector<size_t> m_photonIndexList;

		FW::Mat4f m_projection;
		FW::Mat4f m_worldToCamera;
		FW::Mat4f m_meshScale;

		FW::MulticoreLauncher* m_launcher;
		contextData m_contextData;

		MeshC* m_mesh;
		FW::Mesh<FW::VertexPNC>* m_photonTestMesh;
		FW::Image* m_image;

        FW::GLContext* m_context;
		FW::CameraControls* m_camera;
        AssetManager* m_assetManager;
        FW::GLContext::Program* m_shader;
		
		Node* m_sceneTree;
		Node* m_photonTree;
		bool m_renderWithPhotonMaping;
		bool m_hasPhotonMap;
};