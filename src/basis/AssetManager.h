#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "3d/Mesh.hpp"
#include "gui/Image.hpp"


enum MeshType
{
        MeshType_Pyramid,
		MeshType_Sphere,
		MeshType_Cube,
		MeshType_TestScene,
		MeshType_Cornell,
		MeshType_Pheonix,

        MeshType_Count
};

enum ImageType
{
	ImageType_PhoenexDiffuse,

	ImageType_Count
};

class AssetManager
{
public:
        AssetManager() {}
        ~AssetManager() {}

        void LoadAssets();
        void ReleaseAssets();

        FW::MeshBase* getMesh(MeshType meshType);
		FW::Image* getImage(ImageType type);
		void exportMesh(const std::string& fileName, FW::MeshBase*);

private:
        FW::MeshBase* importMesh(const std::string& fileName);
		FW::Image* importImage(const std::string& fileName);

        FW::MeshBase*	m_meshes[MeshType_Count];
		FW::Image*		m_images[ImageType_Count];

};