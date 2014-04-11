#include "Renderer.h"
#include "RayTracer.h"
#include "3d/Texture.hpp"
#include "Memory.h"
#include "Sampling.h"

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

	m_randomGen = FW::Random();
	m_launcher = new FW::MulticoreLauncher();
	m_launcher->setNumThreads(m_launcher->getNumCores());

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

	RayTracer::get().startUp(m_launcher->getNumCores());
	m_sceneTree = RayTracer::get().constructHierarchy(m_triangles, m_indexListFromScene);
		
}

void Renderer::shutDown()
{
	delete m_mesh;
	delete m_photonTestMesh;
	delete m_launcher;
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
		if(m_launcher->getNumFinished() == m_launcher->getNumTasks())

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
	Node* buffer[1028];

	for(auto i = 0u; i < m_lightSources.size(); ++i)
	{
		size_t s = numOfPhotons * areas[i]/total + 1u;
		float power = m_scanlineContext.d_totalLight/(m_lightSources.size());
		m_triangles[m_lightSources[i]].m_lightPower = power;
		FW::Vec3f photonPower = FW::Vec3f(power)/numOfPhotons; 
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

			//drawTriangleToCamera(org, FW::Vec4f(pow, 1.f));
			
			FW::Vec3f dir = randomVectorToHalfUnitSphere(normal, m_randomGen);
			Hit hit = Hit(10.f);
			RayTracer::get().rayCast(org, dir, hit, m_triangles, m_indexListFromScene, m_sceneTree, buffer);
			
			if(!hit.b)
				continue;

			FW::Vec3f newNormal = (interpolateAttribute(hit.triangle, getDiversion(hit.intersectionPoint,hit.triangle), m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
			newNormal = newNormal.normalized();
			FW::Vec3f pow = power * getAlbedo((TriangleToMeshData*)tri.m_userPointer, m_mesh, div) * FW::dot(newNormal, -dir);

			const TriangleToMeshData* map = (const TriangleToMeshData*) hit.triangle.m_userPointer;
			FW::Vec3f barysHit = FW::Vec3f((1.0f - hit.u - hit.v, hit.u, hit.v));
			FW::Vec3f albedo = getAlbedo(map, m_mesh,barysHit);

			if(shader(hit, m_mesh) == MaterialPM_Lightsource)
			{				
				castIndirectLight(Photon(hit.intersectionPoint, pow, newNormal), hit, buffer);
				continue;
			}
			Photon photon = Photon(hit.intersectionPoint, pow, -(dir.normalized()));
			m_photons.push_back(photon);
			float r3 = randomGen.getF32(0,1.f);
			float threshold = (albedo.x + albedo.y + albedo.z)/3.f;
			if(r3 < threshold)
				castIndirectLight(photon, hit, buffer);
		}
	}
}

void Renderer::castIndirectLight(const Photon& previous, const Hit& hit, Node** buffer)
{
	FW::Random randomGen = FW::Random(m_photons.size());
	const Triangle& tri = hit.triangle;
	const TriangleToMeshData* data = (TriangleToMeshData*) hit.triangle.m_userPointer;
	FW::Vec3f normal = (interpolateAttribute(tri, getDiversion(hit.intersectionPoint,tri), m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
	normal = normal.normalized();

	FW::Vec3f org = previous.pos + 0.001f * normal;
	FW::Vec3f dir = randomVectorToHalfUnitSphere(normal, randomGen);

	Hit h = Hit(10.f);
	RayTracer::get().rayCast(org, dir, h, m_triangles, m_indexListFromScene, m_sceneTree, buffer);
			
	if(!h.b)
		return;

	const TriangleToMeshData* map = (const TriangleToMeshData*) hit.triangle.m_userPointer;
	FW::Vec3f barys = FW::Vec3f((1.0f - h.u - h.v, h.u, h.v));
	const FW::MeshBase::Material& mat = m_mesh->material(map->submeshIndex);
	FW::Vec3f albedo = getAlbedo(map, m_mesh,barys);

	FW::Vec3f newNormal = (interpolateAttribute(h.triangle, getDiversion(h.intersectionPoint,h.triangle), m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
	newNormal = newNormal.normalized();
	FW::Vec3f pow = previous.power * getAlbedo((TriangleToMeshData*)h.triangle.m_userPointer, m_mesh, getDiversion(h.intersectionPoint, h.triangle)) * FW::dot(newNormal, -dir);

	Photon photon = Photon(h.intersectionPoint, pow, -(dir.normalized()));
	m_photons.push_back(photon);
	float threshold = (albedo.x + albedo.y + albedo.z)/3.f;
	float r3 = randomGen.getF32(0,1.0f);
	if(r3 < threshold)
		castIndirectLight(photon, h, buffer);
}

void Renderer::initPhotonMaping(const size_t numOfPhotons, const float r, const size_t FG, const float totalLight, const FW::Vec2i& size)
{
	m_scanlineContext.d_numberOfFGRays=  FG;
	m_scanlineContext.d_FGRadius=  r;
	m_scanlineContext.d_totalLight=  totalLight;
	std::cout << "Starting photon cast..."; 
	m_photons.clear();
	m_photonIndexList.clear();
	m_photonTestMesh->clear();
	m_photonTestMesh->addSubmesh();
	if(m_photonCasted)
		RayTracer::get().demolishTree(m_photonTree);
	castDirectLight(numOfPhotons);
	m_photonCasted = true;
	std::cout << "Photon cast done... " << m_photons.size() << " photons total" << std::endl;
	m_photonTree = RayTracer::get().constructHierarchy(m_photons, m_photonIndexList);
	std::cout << "Creating image... " << std::endl;
	FW::Timer timer;
	timer.start();
	createImage(size);
	std::cout << " done!  Time spend: " << timer.getElapsed() << std::endl;
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
	if(m_renderWithPhotonMaping)
		delete m_image;

	m_image = new FW::Image(size, FW::ImageFormat::RGBA_Vec4f);
	FW::Vec2i imageSize = m_image->getSize();

	FW::Mat4f worldToCamera = m_camera->getWorldToCamera();
	FW::Mat4f projection = FW::Mat4f::fitToView(FW::Vec2f(-1,-1), FW::Vec2f(2,2), imageSize)*m_camera->getCameraToClip();
	FW::Mat4f invP = (projection * worldToCamera).inverted();
	
	m_scanlineContext.d_invP = invP;
	m_scanlineContext.d_vertices = &m_vertices;
	m_scanlineContext.d_triangles = &m_triangles;
	m_scanlineContext.d_triangleToMeshData = &m_triangleToMeshData;
	m_scanlineContext.d_indexListFromScene = &m_indexListFromScene;
	m_scanlineContext.d_photons = &m_photons;
	m_scanlineContext.d_photonIndexList = &m_photonIndexList;
	m_scanlineContext.d_image = m_image;
	m_scanlineContext.d_mesh = m_mesh;
	m_scanlineContext.d_sceneTree = m_sceneTree;
	m_scanlineContext.d_photonTree = m_photonTree;	

	//m_launcher->setNumThreads(1);
	m_launcher->popAll();
	m_launcher->push(imageScanline, &m_scanlineContext, 0, imageSize.y );
	while(m_launcher->getNumFinished() != m_launcher->getNumTasks())
	{
		printf("~ %.2f %% \r", 100.0f*m_launcher->getNumFinished()/(float)m_launcher->getNumTasks());
	}
}

void Renderer::imageScanline(FW::MulticoreLauncher::Task& t)
{
	scanlineData& data = *(scanlineData*)t.data;

	const int y = t.idx;	
	const FW::Vec2f imageSize = data.d_image->getSize();
	Node* buffer[1028];

	for(int x = 0; x < imageSize.x; ++x)
	{	
		const float yP = (y + .5f) / imageSize.y * -2.0f + 1.0f;
		const float xP = (x + .5f) / imageSize.x *  2.0f - 1.0f;
		FW::Vec3f E = FW::Vec3f();
		float totalW = .0f;
		MultiJitteredSamplingWithBoxFilter sampling = MultiJitteredSamplingWithBoxFilter(1, FW::Vec2f(xP, yP), imageSize);
		while(!sampling.isDone())
		{
			float w = .0f;
			FW::Vec2f p = sampling.getNextSamplePos(w);
			FW::Vec4f P0( p.x, p.y, 0.0f, 1.0f );
			FW::Vec4f P1( p.x, p.y, 1.0f, 1.0f );
			FW::Vec4f Roh = (data.d_invP * P0);
			FW::Vec3f Ro = (Roh * (1.0f / Roh.w)).getXYZ();
			FW::Vec4f Rdh = (data.d_invP * P1);
			FW::Vec3f Rd = (Rdh * (1.0f / Rdh.w)).getXYZ();

			Rd = Rd - Ro;
			Hit h = Hit(10.f);
			if(RayTracer::get().rayCast(Ro, Rd, h, *data.d_triangles, *data.d_indexListFromScene, data.d_sceneTree, buffer))
			{
				FW::Vec3f albedo = getAlbedo((TriangleToMeshData*) h.triangle.m_userPointer, data.d_mesh, FW::Vec3f(1.f-h.u-h.v, h.u, h.v));				
				FW::Vec3f dir = (-Rd).normalized();
				if(shader(h, data.d_mesh) == MaterialPM_Lightsource)
				{
					Triangle& tri = h.triangle;
					FW::Vec3f normal = (interpolateAttribute(tri, getDiversion(h.intersectionPoint,tri), data.d_mesh, data.d_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
					normal = normal.normalized();
					float dot = FW::dot(dir, normal);
					if(dot < 0)
						dot = FW::dot(dir, -normal);
				
					E = ((*data.d_triangles)[h.i]).m_lightPower * albedo * dot; 
				}
				else
				{
					E = albedo * gatherPhotons(h,dir,data, buffer);
				}
				E *= w;
				totalW += w;
			}
		}
		E *= 1.f/totalW;
		data.d_image->setVec4f(FW::Vec2i(x,y), FW::Vec4f(E, 1.f));
	}
}

FW::Vec3f Renderer::getAlbedo(const TriangleToMeshData* map, const MeshC* mesh, const FW::Vec3f& barys )
{
	FW::Vec3f Kd;
	const FW::MeshBase::Material& mat = mesh->material(map->submeshIndex);
	if ( mat.textures[FW::MeshBase::TextureType_Diffuse].exists() )
	{
		const FW::Texture& tex = mat.textures[FW::MeshBase::TextureType_Diffuse];
		const FW::Image& teximg = *tex.getImage();
		FW::Vec3f indices = mesh->indices(map->submeshIndex)[map->vertexIndex];

		int attribidx = mesh->findAttrib(FW::MeshBase::AttribType_TexCoord); 
		FW::Vec2f v[3];
		v[0] = mesh->getVertexAttrib( indices[0], attribidx ).getXY();
		v[1] = mesh->getVertexAttrib( indices[1], attribidx ).getXY();
		v[2] = mesh->getVertexAttrib( indices[2], attribidx ).getXY();
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

FW::Vec3f Renderer::gatherPhotons(const Hit& h, const FW::Vec3f& dir, const scanlineData& data, Node** buffer)
{
	const Triangle& tri = h.triangle;
	const TriangleToMeshData* meshData = (TriangleToMeshData*) h.triangle.m_userPointer;
	FW::Vec3f normal = (interpolateAttribute(tri, getDiversion(h.intersectionPoint,tri), data.d_mesh, data.d_mesh->findAttrib(FW::MeshBase::AttribType_Normal))).getXYZ();
	normal = normal.normalized();
	FW::Vec3f org = h.intersectionPoint + 0.001f * normal;
	FW::Vec3f total;
	FW::Random random = FW::Random();
	if(data.d_numberOfFGRays == 1)
	{
		std::vector<HeapNode> nodes;
		float r = data.d_FGRadius;
		RayTracer::get().searchPhotons(h.intersectionPoint, *data.d_photons, *data.d_photonIndexList, data.d_photonTree, r, 10u, nodes, buffer);
		FW::Vec3f E = FW::Vec3f();
		for(auto j = 1u; j < nodes.size(); ++j)
		{
			float dot = FW::dot(dir, (*data.d_photons)[nodes[j].key].dir);
			if(dot < .0f)
				continue;
			E += (*data.d_photons)[nodes[j].key].power * dot;
		}
		return E/(2*r*FW_PI);
	}
	else
	{
		for(auto i = 0u; i < data.d_numberOfFGRays; ++i)
		{			
			FW::Vec3f dir = randomVectorToHalfUnitSphere(normal, random);

			Hit hit = Hit(10.f);
			if(!RayTracer::get().rayCast(org, dir, hit, *data.d_triangles, *data.d_indexListFromScene, data.d_sceneTree, buffer))
				continue;
			std::vector<HeapNode> nodes;
			float r = data.d_FGRadius;
			RayTracer::get().searchPhotons(hit.intersectionPoint, *data.d_photons, *data.d_photonIndexList, data.d_photonTree, r, 10u, nodes, buffer);
			FW::Vec3f E = FW::Vec3f();
			for(auto j = 1u; j < nodes.size(); ++j)
			{
				float dot = FW::dot(-dir, (*data.d_photons)[nodes[j].key].dir);
				if(dot < .0f)
					continue;
				E += (*data.d_photons)[nodes[j].key].power * dot;
			}
			total += E/(2*r*FW_PI);
		}
		total *= 1.f/(float) data.d_numberOfFGRays; 
		return total;
	}
};

MaterialPM Renderer::shader(const Hit& h, MeshC* mesh)
{
	const TriangleToMeshData* data = (TriangleToMeshData*) h.triangle.m_userPointer;
	const FW::MeshBase::Material mat = mesh->material(data->submeshIndex);
	if(mat.specular == FW::Vec3f())
		return  MaterialPM_Lightsource;
	else
		return MaterialPM_Diffuse; 
}

FW::Vec3f Renderer::randomVectorToHalfUnitSphere(const FW::Vec3f& vec, FW::Random& r)
{
		FW::Vec2f rndUnitSquare = r.getVec2f(0.0f,1.0f);
		FW::Vec2f rndUnitDisk = toUnitDisk(rndUnitSquare);
		FW::Mat3f formBasisMat = formBasis(vec);
		FW::Vec3f rndToUnitHalfSphere = FW::Vec3f(rndUnitDisk.x, rndUnitDisk.y, FW::sqrt(1.0f-(rndUnitDisk.x*rndUnitDisk.x)-(rndUnitDisk.y*rndUnitDisk.y)));
		return formBasisMat*rndToUnitHalfSphere;	
}

void Renderer::preCalculateOutgoingLight()
{
	for ( size_t i = 0u; i < m_photons.size(); ++i )
	{
		gatherPhotons(
		if (photonsFound == 0)
			continue;

		RTMaterial* material = m_tempMaterials[i];
		Vec3f normal = m_tempNormals[i];
		RTHit hit = m_tempHits[i];
		const RTToMesh* map = (const RTToMesh*)hit.triangle->m_userPointer;

		float alpha = 10.818f;
		float beta = 1.953f;
		float e = 2.718281f;
		float rMax = sqrt(knnPhotons[1].dst);

		Vec3f Li = Vec3f();
		for ( int j = 1; j <= photonsFound; ++j )
		{		
			Vec3f lightDir = knnPhotons[j].photon->dir;
			if ( normal.dot( lightDir ) > 0 )
			{
				float d = sqrt(knnPhotons[j].dst);
				float t1 = 1.0f - pow( e, ( -beta * d * d / ( 2.0f * rMax * rMax) ) );
				float t2 = 1.0f - pow( e, -beta );
				float w = alpha * ( 1.0f - (t1 / t2) );
				Li += material->shade( normal, lightDir, normal, knnPhotons[j].photon->E, map, hit.barys ) * w;
			}
		}
		float A = FW_PI*rMax*rMax;
		radianceMap->m_photonMap[i].E = Li / A * FW_PI;
	}
}