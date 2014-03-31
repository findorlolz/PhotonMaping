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

        MeshType_Count
};

class AssetManager
{
public:
        AssetManager() {}
        ~AssetManager() {}

        void LoadAssets();
        void ReleaseAssets();

        FW::MeshBase* getMesh(MeshType meshType);
		void exportMesh(const const std::string& fileName, FW::MeshBase*);

private:
        FW::MeshBase* importMesh(const std::string& fileName);

        FW::MeshBase* m_meshes[MeshType_Count];

};