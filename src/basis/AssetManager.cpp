#include "AssetManager.h"
#include "io/ImageLodePngIO.hpp"
#include "io/File.hpp"


void AssetManager::LoadAssets()
{
		m_meshes[MeshType_Cube] = importMesh("lightsourcecube.obj");
		m_meshes[MeshType_CornellSphere] = importMesh("cornellsphere.obj");
		m_meshes[MeshType_Pheonix] = importMesh("finaltest.obj");
		m_meshes[MeshType_CornellTest] = importMesh("cornellcubetest.obj");
}

void AssetManager::ReleaseAssets()
{
        for (auto mesh : m_meshes)
                delete mesh;
}

FW::MeshBase* AssetManager::getMesh(MeshType meshType)
{
        if (meshType == MeshType_Count)
                return NULL;
        return m_meshes[meshType];
}

FW::Image* AssetManager::getImage(ImageType imageType)
{
        if (imageType == MeshType_Count)
                return NULL;
		return m_images[imageType];
}

FW::MeshBase* AssetManager::importMesh(const std::string& fileName)
{
        std::string filePath = "assets/meshes/" + fileName;
        return FW::importMesh(filePath.c_str());
}

void AssetManager::exportMesh(const const std::string& fileName, FW::MeshBase* mesh)
{
	std::string filePath = "assets/meshes/export/" + fileName;
    return FW::exportMesh(filePath.c_str(), mesh);
}

FW::Image* AssetManager::importImage(const std::string& fileName)
{
	std::string filePath = "assets/images/" + fileName;
	FW::File guiImage(filePath.c_str(), FW::File::Read);
	return FW::importLodePngImage(guiImage);
}