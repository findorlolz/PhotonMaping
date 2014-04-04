#pragma once

#include "AssetManager.h"
#include "3d/CameraControls.hpp"
#include "HelpFunctions.h"


typedef FW::Mesh<FW::VertexPNTC> MeshC;

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
		void initPhotonMaping(const size_t, const FW::Vec2i&);
		void clearTriangles();
		void toggleRenderingMode() { m_renderWithPhotonMaping = !m_renderWithPhotonMaping; }

private:
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
		bool m_photonCasted;

		 Renderer() {}
        ~Renderer() {}

		void castIndirectLight(const Photon&, const Hit& hit);
		void castDirectLight(const size_t, const float);

		void initTrianglesFromMesh(MeshType, const FW::Vec3f&);
		void updateTriangleToMeshDataPointers();

		void drawTriangleToCamera(const FW::Vec3f& pos, const FW::Vec4f& color);
		void drawPhotonMap();

		void createImage(const FW::Vec2i& size);

		FW::Vec4f interpolateAttribute(const Triangle& tri, const FW::Vec3f&, const FW::MeshBase* mesh, int attribidx);
		FW::Vec3f getDiversion(const FW::Vec3f&, const Triangle&);
		FW::Vec3f getAlbedo(const TriangleToMeshData*, const FW::Vec3f&);
		FW::Vec3f gatherPhotons(const Hit&, const size_t, const float);
};