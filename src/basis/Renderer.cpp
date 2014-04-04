#include "Renderer.h"
#include "RayTracer.h"
#include "3d/Texture.hpp"
#include "Memory.h"

void Renderer::startUp(FW::GLContext* gl, FW::CameraControls* camera, AssetManager* assetManager)
{
	m_context = gl; 
    m_assetManager = assetManager;
	m_camera = camera;
	m_projection = FW::Mat4f();
	m_worldToCamera = FW::Mat4f();
	m_camera->setPosition(FW::Vec3f(.0f, .8f, .8f));
	m_camera->setForward(FW::Vec3f(.0f, -1.f, -1.f));
	m_camera->setFar(20.0f);
	m_camera->setSpeed(4.0f);

	m_renderWithPhotonMaping = false;
	m_photonCasted = false;

	m_mesh = new MeshC();
	m_mesh->append(*(m_assetManager->getMesh(MeshType_Cornell)));
	m_mesh->append(*m_assetManager->getMesh(MeshType_Pheonix));
	m_photonTestMesh = new FW::Mesh<FW::VertexPNC>();
	m_photonTestMesh->addSubmesh();
	updateTriangleToMeshDataPointers();

	for (auto i = 0u; i < m_mesh->numVerticesU(); ++i )
		m_mesh->mutableVertex(i).c = FW::Vec3f(1,1,1);

	RayTracer::get().startUp();
	m_sceneTree = RayTracer::get().constructHierarchy(m_triangles, m_indexListFromScene);
		
}

void Renderer::shutDown()
{
	delete m_mesh;
	delete m_photonTestMesh;
	RayTracer::get().demolishTree(m_sceneTree);
	if(m_photonCasted)
	{
		RayTracer::get().demolishTree(m_photonTree);
		delete m_image;
	}
	RayTracer::get().shutDown();
	delete &get();
}

void Renderer::drawFrame()
{
	glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);	
	if(m_renderWithPhotonMaping)
	{
		m_context->drawImage(*m_image, FW::Vec2f());
		return;
	}
	else
	{
		m_projection = m_context->xformFitToView(FW::Vec2f(-1.0f, -1.0f), FW::Vec2f(2.0f, 2.0f)) * m_camera->getCameraToClip();
		m_worldToCamera = m_camera->getWorldToCamera();
		m_mesh->draw(m_context, m_worldToCamera, m_projection);
		if(m_photonCasted)
			drawPhotonMap();
	}
	glDrawBuffer(GL_BACK);
	glBindVertexArray(0);
	glUseProgram(0);
}

void Renderer::clearTriangles()
{
	m_vertices.clear();
	m_triangles.clear();
	m_triangleToMeshData.clear();
}

void Renderer::updateTriangleToMeshDataPointers()
{
	for (size_t i = 0u; i < m_mesh->numVerticesU(); ++i )
	{
		FW::Vec3f p = m_mesh->getVertexAttrib(i, FW::MeshBase::AttribType_Position).getXYZ();
		m_vertices.push_back(p);
	}
	
	for (size_t i = 0u; i < m_mesh->numSubmeshes(); ++i )
	{
		const FW::Array<FW::Vec3i>& idx = m_mesh->indices(i);
		const FW::MeshBase::Material mat = m_mesh->material(i);
		for (size_t j = 0u; j < idx.getSize(); ++j )
		{
			TriangleToMeshData m;
			m.submeshIndex = i;
			m.vertexIndex = j;
			m_triangleToMeshData.push_back(m);

			Triangle t;
			t.m_userPointer = 0;
			t.m_vertices[0] = &m_vertices[0] + idx[j][0];
			t.m_vertices[1] = &m_vertices[0] + idx[j][1];
			t.m_vertices[2] = &m_vertices[0] + idx[j][2];
			if(mat.specular == FW::Vec3f())
			{
				t.m_lightPower = 1.f;
				m_lightSources.push_back(m_triangles.size());
			}
			m_triangles.push_back(t);
		}
	}	
	
	for ( size_t i = 0; i < m_triangles.size(); ++i )
		m_triangles[ i ].m_userPointer = &m_triangleToMeshData[i];
}

void Renderer::castDirectLight(const size_t numOfPhotons, const float maxPower)
{
	std::vector<float> areas = std::vector<float> (m_lightSources.size());
	float total = .0f;
	for(auto i = 0u; i < m_lightSources.size(); ++i)
	{
		float tmp = .5f * (FW::cross((*(m_triangles[m_lightSources[i]].m_vertices[1]) - *(m_triangles[m_lightSources[i]].m_vertices[0])), (*(m_triangles[m_lightSources[i]].m_vertices[2]) - *(m_triangles[m_lightSources[i]].m_vertices[0])))).length();		
		areas[i] = tmp;
		total += tmp;
	}

	FW::Random randomGen = FW::Random();
	for(auto i = 0u; i < m_lightSources.size(); ++i)
	{
		size_t s = numOfPhotons * areas[i]/total + 1u;
		FW::Vec3f power = m_triangles[m_lightSources[i]].m_lightPower * maxPower;
		for(auto j = 0u; j < s; ++j)
		{
			const Triangle& tri = m_triangles[m_lightSources[i]];
			FW::Vec3f A = *tri.m_vertices[0];
			FW::Vec3f B = *tri.m_vertices[1];
			FW::Vec3f C = *tri.m_vertices[2];
		
			float sqr_r1 = FW::sqrt(randomGen.getF32(0,1.0f));
			float r2 = randomGen.getF32(0,1.0f);
			FW::Vec3f org = (1-sqr_r1)*A + sqr_r1*(1-r2)*B + sqr_r1*r2*C;
			FW::Vec3f div = getDiversion(org,tri);
			FW::Vec3f normal = (interpolateAttribute(tri, div, m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
			normal = normal.normalized();
			FW::Vec3f pow = power * getAlbedo((TriangleToMeshData*)tri.m_userPointer, div);

			//drawTriangleToCamera(org, FW::Vec4f(pow, 1.f));
			
			FW::Vec2f rndUnitSquare = randomGen.getVec2f(0.0f,1.0f);
			FW::Vec2f rndUnitDisk = toUnitDisk(rndUnitSquare);
			FW::Mat3f formBasisMat = formBasis(normal);
			FW::Vec3f rndToUnitHalfSphere = FW::Vec3f(rndUnitDisk.x, rndUnitDisk.y, FW::sqrt(1.0f-(rndUnitDisk.x*rndUnitDisk.x)-(rndUnitDisk.y*rndUnitDisk.y)));
			FW::Vec3f dir = formBasisMat*rndToUnitHalfSphere;
			Hit hit = Hit(10.f);
			RayTracer::get().rayCast(org, dir, hit, m_triangles, m_indexListFromScene, m_sceneTree);
			
			if(!hit.b)
				continue;

			const TriangleToMeshData* map = (const TriangleToMeshData*) hit.triangle.m_userPointer;
			FW::Vec3f barysHit = FW::Vec3f((1.0f - hit.u - hit.v, hit.u, hit.v));
			const FW::MeshBase::Material& mat = m_mesh->material(map->submeshIndex);
			FW::Vec3f albedo = getAlbedo(map, barysHit);

			Photon photon = Photon(hit.intersectionPoint, albedo*power, -(dir.normalized()));
			m_photons.push_back(photon);
			float r3 = randomGen.getF32(0,1.0f);
			float threshold = (albedo.x + albedo.y + albedo.z)/3.f;
			if(r3 > threshold)
				castIndirectLight(photon, hit);
		}
	}
}

void Renderer::castIndirectLight(const Photon& previous, const Hit& hit)
{
	FW::Random randomGen = FW::Random();
	const Triangle& tri = hit.triangle;
	const TriangleToMeshData* data = (TriangleToMeshData*) hit.triangle.m_userPointer;
	FW::Vec3f normal = (interpolateAttribute(tri, getDiversion(hit.intersectionPoint,tri), m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
	normal = normal.normalized();

	FW::Vec3f org = previous.pos + 0.001f * normal;
	//drawTriangleToCamera(org, FW::Vec4f(previous.power, 1.f));

	FW::Vec2f rndUnitSquare = randomGen.getVec2f(0.0f,1.0f);
	FW::Vec2f rndUnitDisk = toUnitDisk(rndUnitSquare);
	FW::Mat3f formBasisMat = formBasis(normal);
	FW::Vec3f rndToUnitHalfSphere = FW::Vec3f(rndUnitDisk.x, rndUnitDisk.y, FW::sqrt(1.0f-(rndUnitDisk.x*rndUnitDisk.x)-(rndUnitDisk.y*rndUnitDisk.y)));
	FW::Vec3f dir = formBasisMat*rndToUnitHalfSphere;

	Hit h = Hit(10.f);
	RayTracer::get().rayCast(org, dir, h, m_triangles, m_indexListFromScene, m_sceneTree);
			
	if(!h.b)
		return;

	const TriangleToMeshData* map = (const TriangleToMeshData*) hit.triangle.m_userPointer;
	FW::Vec3f barys = FW::Vec3f((1.0f - hit.u - hit.v, hit.u, hit.v));
	const FW::MeshBase::Material& mat = m_mesh->material(map->submeshIndex);
	FW::Vec3f albedo = getAlbedo(map, barys);

	FW::Vec3f pow = previous.power * albedo;
	Photon photon = Photon(h.intersectionPoint, pow, -(dir.normalized()));
	m_photons.push_back(photon);
	float threshold = (albedo.x + albedo.y + albedo.z)/3.f;
	float r3 = randomGen.getF32(0,1.0f);
	if(r3 < threshold)
		castIndirectLight(photon, h);
}

void Renderer::initPhotonMaping(const size_t numOfPhotons, const FW::Vec2i& size)
{
	std::cout << "Starting photon cast..."; 
	m_photons.clear();
	m_photonTestMesh->clear();
	m_photonTestMesh->addSubmesh();
	if(m_photonCasted)
		RayTracer::get().demolishTree(m_photonTree);
	castDirectLight(numOfPhotons, .5f);
	m_photonCasted = true;
	std::cout << "Photon cast done... " << m_photons.size() << " photons total" << std::endl;
	m_photonTree = RayTracer::get().constructHierarchy(m_photons, m_photonIndexList);
	std::cout << "Create image... ";
	FW::Timer timer;
	timer.start();
	createImage(size);
	std::cout << " done! Time spend: " << timer.getElapsed() << std::endl;
	m_renderWithPhotonMaping = true;
}

void Renderer::drawPhotonMap()
{
/*	m_photonTestMesh->draw(m_context, m_worldToCamera, m_projection);
	glLineWidth(2.f);
	glBegin(GL_LINES);
	float l = .1f;
	for (int i = 0; i < m_photons.size(); ++i )
	{
		FW::Vec3f c = FW::Vec3f(1.f, .0f, .0f);
		glColor3fv(&c.x);
		glVertex3f( m_photons[i].pos.x, m_photons[i].pos.y, m_photons[i].pos.z );
		glVertex3f( m_photons[i].previouspos.x, m_photons[i].previouspos.y, m_photons[i].previouspos.z );
	}
	glEnd();*/
}

void Renderer::drawTriangleToCamera(const FW::Vec3f& pos, const FW::Vec4f& color)
{
	const float particleSize = .003;
	FW::Vec3f n = (m_camera->getPosition() - pos).normalized();
	FW::Vec3f t = n.cross(m_camera->getUp());
	FW::Vec3f p = pos + 0.00001f * n;
	FW::Vec3f b = n.cross(t);

	FW::VertexPNC vertexArray[] =
    {
		FW::VertexPNC((p + t * particleSize + b * particleSize), n, color),
        FW::VertexPNC((p - t * particleSize + b * particleSize), n, color),
		FW::VertexPNC((p + t * particleSize - b * particleSize), n, color),
		FW::VertexPNC((p - t * particleSize - b * particleSize), n, color)
	};

	static const FW::Vec3i indexArray[] =
    {
        FW::Vec3i(0,1,2),  FW::Vec3i(1,3,2)
    };

	int base = m_photonTestMesh->numVertices();
    m_photonTestMesh->addVertices(vertexArray, FW_ARRAY_SIZE(vertexArray));

    FW::Array<FW::Vec3i>& indices = m_photonTestMesh->mutableIndices(0);
    for (int i = 0; i < (int)FW_ARRAY_SIZE(indexArray); i++)
        indices.add(indexArray[i] + base);
}

FW::Vec4f Renderer::interpolateAttribute(const Triangle& tri, const FW::Vec3f& div, const FW::MeshBase* mesh, int attribidx )
{
	const TriangleToMeshData* map = (const TriangleToMeshData*)tri.m_userPointer;

	FW::Vec4f v[3];
	v[0] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][0], attribidx );
	v[1] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][1], attribidx );
	v[2] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][2], attribidx );
	return div.x*v[0] + div.y*v[1] + div.z*v[2];
}

FW::Vec3f Renderer::getDiversion(const FW::Vec3f& p, const Triangle& tri)
{
	float A = (*tri.m_vertices[0] - p).length();
	float B = (*tri.m_vertices[1] - p).length();
	float C = (*tri.m_vertices[2] - p).length();
	float total = A + B + C;

	return FW::Vec3f(A/total, B/total, C/total);
}

void Renderer::createImage(const FW::Vec2i& size)
{
	if(!m_photonCasted)
		return;

	m_image = new FW::Image(size, FW::ImageFormat::RGBA_Vec4f);
	FW::Vec2i imageSize = m_image->getSize();

	FW::Mat4f worldToCamera = m_camera->getWorldToCamera();
	FW::Mat4f projection = FW::Mat4f::fitToView(FW::Vec2f(-1,-1), FW::Vec2f(2,2), imageSize)*m_camera->getCameraToClip();
	FW::Mat4f invP = (projection * worldToCamera).inverted();
	
	for(int x = 0; x < imageSize.x; ++x)
	{	
		#pragma omp parallel for
		for(int y = 0; y < imageSize.y; ++y)
		{
			float xP = (x + .5f) / imageSize.x *  2.0f - 1.0f;
			float yP = (y + .5f) / imageSize.y * -2.0f + 1.0f;
			FW::Vec4f P0( xP, yP, 0.0f, 1.0f );
			FW::Vec4f P1( xP, yP, 1.0f, 1.0f );
			FW::Vec4f Roh = (invP * P0);
			FW::Vec3f Ro = (Roh * (1.0f / Roh.w)).getXYZ();
			FW::Vec4f Rdh = (invP * P1);
			FW::Vec3f Rd = (Rdh * (1.0f / Rdh.w)).getXYZ();


			Rd = Rd - Ro;
			FW::Vec3f E = FW::Vec3f();
			Hit h = Hit(10.f);
			if(RayTracer::get().rayCast(Ro, Rd, h, m_triangles, m_indexListFromScene, m_sceneTree))
			{
				FW::Vec3f albedo = getAlbedo((TriangleToMeshData*) h.triangle.m_userPointer, FW::Vec3f(1.f-h.u-h.v, h.u, h.v));
				E = albedo * gatherPhotons(h, 10, .1);
			}
			m_image->setVec4f(FW::Vec2i(x,y), FW::Vec4f(E, 1.f));
		}
	}
}

FW::Vec3f Renderer::getAlbedo(const TriangleToMeshData* map, const FW::Vec3f& barys )
{
	FW::Vec3f Kd;
	const FW::MeshBase::Material& mat = m_mesh->material(map->submeshIndex);
	if ( mat.textures[FW::MeshBase::TextureType_Diffuse].exists() )
	{
		const FW::Texture& tex = mat.textures[FW::MeshBase::TextureType_Diffuse];
		const FW::Image& teximg = *tex.getImage();
		FW::Vec3f indices = m_mesh->indices(map->submeshIndex)[map->vertexIndex];

		int attribidx = m_mesh->findAttrib(FW::MeshBase::AttribType_TexCoord); 
		FW::Vec2f v[3];
		v[0] = m_mesh->getVertexAttrib( indices[0], attribidx ).getXY();
		v[1] = m_mesh->getVertexAttrib( indices[1], attribidx ).getXY();
		v[2] = m_mesh->getVertexAttrib( indices[2], attribidx ).getXY();
		FW::Vec2f UV = barys[0]*v[0] + barys[1]*v[1] + barys[2]*v[2];

		FW::Vec2i imageSize = teximg.getSize();

		int x = UV.x * imageSize.x;
		while(x < 0)
			x += (imageSize.x-1);

		x = x % (imageSize.x - 1);

		int y = UV.y * imageSize.y;
		while(x < 0)
			y += (imageSize.x-1);

		y = y % (imageSize.x - 1);
					

		FW::Vec2i pos = FW::Vec2i(x,y);

		FW::Vec3f color = teximg.getVec4f(pos).getXYZ();
		Kd = color;
	}
	else
		Kd = mat.diffuse.getXYZ();

	return Kd;
}

FW::Vec3f Renderer::gatherPhotons(const Hit& h, const size_t numOfFGRays, const float r)
{
	FW::Random randomGen = FW::Random();
	const Triangle& tri = h.triangle;
	const TriangleToMeshData* data = (TriangleToMeshData*) h.triangle.m_userPointer;
	FW::Vec3f normal = (interpolateAttribute(tri, getDiversion(h.intersectionPoint,tri), m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
	normal = normal.normalized();
	FW::Vec3f org = h.intersectionPoint + 0.001f * normal;
	FW::Vec3f total;

	for(auto i = 0u; i < numOfFGRays; ++i)
	{
		FW::Vec2f rndUnitSquare = randomGen.getVec2f(0.0f,1.0f);
		FW::Vec2f rndUnitDisk = toUnitDisk(rndUnitSquare);
		FW::Mat3f formBasisMat = formBasis(normal);
		FW::Vec3f rndToUnitHalfSphere = FW::Vec3f(rndUnitDisk.x, rndUnitDisk.y, FW::sqrt(1.0f-(rndUnitDisk.x*rndUnitDisk.x)-(rndUnitDisk.y*rndUnitDisk.y)));
		FW::Vec3f dir = formBasisMat*rndToUnitHalfSphere;

		Hit hit = Hit(10.f);
		if(!RayTracer::get().rayCast(org, dir, hit, m_triangles, m_indexListFromScene, m_sceneTree))
			continue;
		std::vector<HeapNode> nodes;
		float r = .4f;
		RayTracer::get().searchPhotons(hit.intersectionPoint, m_photons, m_photonIndexList, m_photonTree, r, 100u, nodes);
		FW::Vec3f E = FW::Vec3f();
		for(auto j = 1u; j < nodes.size(); ++j)
		{
			float dot = FW::dot(-dir, m_photons[nodes[j].key].dir);
			if(dot < .0f)
				continue;
			E += m_photons[nodes[j].key].power * dot;
		}
		total += E/(2*FW_PI*r);
	}
	total *= 1.f/(float) numOfFGRays; 
	return total;
};

