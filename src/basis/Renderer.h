#pragma once

#include "AssetManager.h"
#include "3d/CameraControls.hpp"
#include "HelpFunctions.h"


typedef FW::Mesh<FW::VertexPNC> MeshC;

class RayTracer;

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
		void initPhotonMaping(const size_t);
		void clearTriangles();

private:
		std::vector<FW::Vec3f> m_vertices;
		std::vector<Triangle> m_triangles;
		std::vector<size_t> m_lightSources;
		std::vector<TriangleToMeshData> m_triangleToMeshData;
		std::vector<size_t> m_indexListFromScene;
		std::vector<Photon> m_photons;

		FW::Mat4f m_projection;
		FW::Mat4f m_worldToCamera;
		FW::Mat4f m_meshScale;

		MeshC* m_mesh;
		MeshC* m_photonTestMesh;
		FW::Image* m_image;

        FW::GLContext* m_context;
		FW::CameraControls* m_camera;
        AssetManager* m_assetManager;
        FW::GLContext::Program* m_shader;
		
		Node* m_sceneTree;
		bool m_renderWithPhotonMaping;
		bool m_photonCasted;

		 Renderer() {}
        ~Renderer() {}

		void castIndirectLight(const Photon&, const Hit& hit);
		void castDirectLight(const size_t);

		void initTrianglesFromMesh(MeshType, const FW::Vec3f&);
		void updateTriangleToMeshDataPointers();

		void drawTriangleToCamera(const FW::Vec3f& pos, const FW::Vec4f& color);
		void drawPhotonMap();
};