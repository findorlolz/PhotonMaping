#include "AssetManager.h"


void AssetManager::LoadAssets()
{
        m_meshes[MeshType_Pyramid] = importMesh("pyramid.obj");
		m_meshes[MeshType_Sphere] = importMesh("sphere.obj"); 
		m_meshes[MeshType_Cube] = importMesh("cube.obj");
		m_meshes[MeshType_TestScene] = importMesh("testscene.obj");
		m_meshes[MeshType_Cornell] = importMesh("cornell.obj");
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