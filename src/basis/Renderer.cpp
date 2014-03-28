#include "Renderer.h"
#include "RayTracer.h"

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
	m_mesh->append(*(m_assetManager->getMesh(MeshType_TestScene)));
	m_mesh->append(*(m_assetManager->getMesh(MeshType_Cube)));
	m_photonTestMesh = new MeshC();
	m_photonTestMesh->addSubmesh();
	updateTriangleToMeshDataPointers();

	for (auto i = 0u; i < m_mesh->numVertices(); ++i )
		m_mesh->mutableVertex(i).c = FW::Vec4f(1,1,1,1);

	RayTracer::get().startUp();
	m_sceneTree = RayTracer::get().constructHierarchy(m_triangles, m_indexListFromScene);
		
}

void Renderer::shutDown()
{
	delete m_mesh;
	delete m_photonTestMesh;
	RayTracer::get().demolishTree(m_sceneTree);
	RayTracer::get().shutDown();
	delete &get();
}

void Renderer::drawFrame()
{
	if(m_renderWithPhotonMaping)
	{

	}
	else
	{
		glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		m_projection = m_context->xformFitToView(FW::Vec2f(-1.0f, -1.0f), FW::Vec2f(2.0f, 2.0f)) * m_camera->getCameraToClip();
		m_worldToCamera = m_camera->getWorldToCamera();
		m_mesh->draw(m_context, m_worldToCamera, m_projection);
		if(m_photonCasted)
			drawPhotonMap();
		glDrawBuffer(GL_BACK);
		glBindVertexArray(0);
		glUseProgram(0);
	}

}

void Renderer::clearTriangles()
{
	m_vertices.clear();
	m_triangles.clear();
	m_triangleToMeshData.clear();
}

void Renderer::updateTriangleToMeshDataPointers()
{
	for (size_t i = 0u; i < m_mesh->numVertices(); ++i )
	{
		FW::Vec3f p = m_mesh->getVertexAttrib(i, FW::MeshBase::AttribType_Position).getXYZ();
		m_vertices.push_back(p);
		p.print();
	}
	
	for (size_t i = 0u; i < m_mesh->numSubmeshes(); ++i )
	{
		const FW::Array<FW::Vec3i>& idx = m_mesh->indices(i);
		for ( int j = 0; j < idx.getSize(); ++j )
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
			m_triangles.push_back(t);
			m_lightSources.push_back(m_triangles.size() - 1);
		}
	}	
	
	for ( size_t i = 0; i < m_triangles.size(); ++i )
		m_triangles[ i ].m_userPointer = &m_triangleToMeshData[i];
}

void Renderer::castDirectLight(const size_t numOfPhotons)
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
		FW::Vec3f power = m_triangles[m_lightSources[i]].m_lightPower * 1.f/(float) s;
		for(auto j = 0u; j < s; ++j)
		{
			FW::Vec3f A = *m_triangles[m_lightSources[i]].m_vertices[0];
			FW::Vec3f B = *m_triangles[m_lightSources[i]].m_vertices[1];
			FW::Vec3f C = *m_triangles[m_lightSources[i]].m_vertices[2];
			FW::Vec3f normal = FW::cross(B-A, C-A).normalized();
		
			float sqr_r1 = FW::sqrt(randomGen.getF32(0,1.0f));
			float r2 = randomGen.getF32(0,1.0f);
			FW::Vec3f org = (1-sqr_r1)*A + sqr_r1*(1-r2)*B + sqr_r1*r2*C;
			drawTriangleToCamera(org, FW::Vec4f(1.f, 1.f, .0f, 1.f));
			
			FW::Vec2f rndUnitSquare = randomGen.getVec2f(0.0f,1.0f);
			FW::Vec2f rndUnitDisk = toUnitDisk(rndUnitSquare);
			FW::Mat3f formBasisMat = formBasis(normal);
			FW::Vec3f rndToUnitHalfSphere = FW::Vec3f(rndUnitDisk.x, rndUnitDisk.y, FW::sqrt(1.0f-(rndUnitDisk.x*rndUnitDisk.x)-(rndUnitDisk.y*rndUnitDisk.y)));
			FW::Vec3f dir = formBasisMat*rndToUnitHalfSphere;
			Hit hit = Hit(10.f);
			RayTracer::get().rayCast(org, dir, hit, m_triangles, m_indexListFromScene, m_sceneTree);
			
			if(!hit.b)
				continue;

			Photon photon = Photon(hit.intersectionPoint, power, dir, org);
			m_photons.push_back(photon);
			float r3 = randomGen.getF32(0,1.0f);
			if(r3 > .5f)
				castIndirectLight(photon, hit);
		}
	}
}

void Renderer::castIndirectLight(const Photon& previous, const Hit& hit)
{
	FW::Random randomGen = FW::Random();

	FW::Vec3f normal = FW::cross((*(hit.triangle.m_vertices[1]) - *(hit.triangle.m_vertices[0])), (*(hit.triangle.m_vertices[2]) - *(hit.triangle.m_vertices[0])));
	normal = normal.normalized();
	if(FW::dot(normal, previous.dir) < 0)
		normal *= -1.f;

	FW::Vec3f L = previous.dir * -1.f;
	FW::Vec3f mirror = normal * 2 * FW::dot(normal, L) - L;
	FW::Vec3f org = previous.pos + 0.001f * normal;

	FW::Vec2f rndUnitSquare = randomGen.getVec2f(0.0f,1.0f);
	FW::Vec2f rndUnitDisk = toUnitDisk(rndUnitSquare);
	FW::Mat3f formBasisMat = formBasis(mirror);
	FW::Vec3f rndToUnitHalfSphere = FW::Vec3f(rndUnitDisk.x, rndUnitDisk.y, FW::sqrt(1.0f-(rndUnitDisk.x*rndUnitDisk.x)-(rndUnitDisk.y*rndUnitDisk.y)));
	FW::Vec3f dir = formBasisMat*rndToUnitHalfSphere;

	/*
	const TriangleToMeshData* map = (const TriangleToMeshData*) hit.triangle.m_userPointer;
	FW::Vec3f barys = FW::Vec3f((1.0f - hit.u - hit.v, hit.u, hit.v));
	const FW::MeshBase::Material& mat = m_mesh->material(map->submesh);
	FW::Vec3f BRDF = m_mesh->material(map->submeshIndex);
	*/

	FW::Vec3f power = previous.power;

	Hit h = Hit(10.f);
	RayTracer::get().rayCast(org, dir, h, m_triangles, m_indexListFromScene, m_sceneTree);
			
	if(!h.b)
		return;

	Photon photon = Photon(h.intersectionPoint, power, dir, org);
	m_photons.push_back(photon);
	float r3 = randomGen.getF32(0,1.0f);
	/*if(r3 > .5f)
		castIndirectLight(photon, h);*/
}

void Renderer::initPhotonMaping(const size_t numOfPhotons)
{
	std::cout << "Starting photon cast..."; 
	m_photons.clear();
	castDirectLight(numOfPhotons);
	m_photonCasted = true;
	std::cout << "Photon cast done... " << m_photons.size() << " photons total" << std::endl;
}

void Renderer::drawPhotonMap()
{
	m_photonTestMesh->draw(m_context, m_worldToCamera, m_projection);
	glLineWidth(.75f);
	glBegin(GL_LINES);
	FW::Vec3f c;
	glColor3fv(&c.x);
	for (int i = 0; i < m_photons.size(); ++i )
	{
		glVertex3f( m_photons[i].pos.x, m_photons[i].pos.y, m_photons[i].pos.z );
		glVertex3f( m_photons[i].previouspos.x, m_photons[i].previouspos.y, m_photons[i].previouspos.z );
	}
	glEnd();
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