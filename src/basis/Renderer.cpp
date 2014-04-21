#include "Renderer.h"
#include "RayTracer.h"
#include "3d/Texture.hpp"
#include "Memory.h"
#include "Sampling.h"

void Renderer::startUp(FW::GLContext* gl, FW::CameraControls* camera, AssetManager* assetManager)
{
	std::cout << "Starting up the Renderer..." << std::endl;
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
	m_hasPhotonMap = false;

	m_mesh = new MeshC();
	m_mesh->append(*(m_assetManager->getMesh(MeshType_CornellTest)));

	m_mesh->collapseVertices();

	m_photonTestMesh = new FW::Mesh<FW::VertexPNC>();
	m_photonTestMesh->addSubmesh();
	updateTriangleToMeshDataPointers();

	for (auto i = 0u; i < m_mesh->numVerticesU(); ++i )
		m_mesh->mutableVertex(i).c = FW::Vec3f(1,1,1);

	RayTracer::get().startUp();
	m_sceneTree = RayTracer::get().constructHierarchy(m_triangles, m_indexListFromScene);
	std::cout << std::endl;
	std::cout << "Welcome to the most EPIC PhotonMapingEexperience! Have a nice ride <-(0_o)/~" << std::endl;
	std::cout << std::endl;
	std::cout << "///////////////////////////////////////////////" << std::endl;
	std::cout << "Press 1 for casting photons and image synthesis" << std::endl;
	std::cout << "Press 2 for image synthesis from existing PM" << std::endl;
	std::cout << "Press 3 to toggle between rendering options" << std::endl;
	std::cout << "Press 4 to toggle visibility of PM parameters" << std::endl;
	std::cout << "//////////////////////////////////////////////" << std::endl;
	std::cout << std::endl;
}

void Renderer::shutDown()
{
	delete m_mesh;
	delete m_photonTestMesh;
	delete m_launcher;
	RayTracer::get().demolishTree(m_sceneTree);
	if(m_hasPhotonMap)
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

	FW::Vec3f lightPosApp = FW::Vec3f();
	for (size_t i = 0u; i < m_mesh->numSubmeshes(); ++i )
	{
		const FW::Array<FW::Vec3i>& idx = m_mesh->indices(i);
		FW::MeshBase::Material* mat = &(m_mesh->material(i));
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
			if(mat->emissive != FW::Vec3f())
			{
				t.m_lightPower = &(mat->emissive);
				m_lightSources.push_back(m_triangles.size());
				lightPosApp += (*(t.m_vertices[0])+*(t.m_vertices[1])+*(t.m_vertices[2]))/3.f;
			}
			m_triangles.push_back(t);
		}
	}

	lightPosApp *= 1.f/(float)m_lightSources.size();
	m_contextData.d_lightPosEstimate = lightPosApp;

	for ( size_t i = 0; i < m_triangles.size(); ++i )
		m_triangles[ i ].m_userPointer = &m_triangleToMeshData[i];
}

void Renderer::castPhotons(const size_t numOfPhotons, std::vector<Hit>& tmpHitList)
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

	FW::Vec3f photonPower = FW::Vec3f(m_contextData.d_totalLight)/numOfPhotons; 

	for(auto i = 0u; i < m_lightSources.size(); ++i)
	{
		size_t s = numOfPhotons * areas[i]/total + 1u;
		const FW::Vec3f& lightSourcePower = *(m_triangles[m_lightSources[i]].m_lightPower);
		const Triangle& tri = m_triangles[m_lightSources[i]];
		FW::Vec3f A = *tri.m_vertices[0];
		FW::Vec3f B = *tri.m_vertices[1];
		FW::Vec3f C = *tri.m_vertices[2];

		for(auto j = 0u; j < s; ++j)
		{	
			float sqr_r1 = FW::sqrt(randomGen.getF32(0,1.0f));
			float r2 = randomGen.getF32(0,1.0f);
			FW::Vec3f orig = (1-sqr_r1)*A + sqr_r1*(1-r2)*B + sqr_r1*r2*C;
			FW::Vec3f normalLightSource = (interpolateAttribute(tri, orig, m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal)));
			normalLightSource = normalLightSource.normalized();
			FW::Vec3f albedoLightSource = getAlbedo((TriangleToMeshData*)tri.m_userPointer, m_mesh,getBarys(tri, orig, m_mesh)); 
			
			FW::Vec3f dir = randomVectorToHalfUnitSphere(normalLightSource, m_randomGen);
			FW::Vec3f E = lightSourcePower * photonPower * albedoLightSource;

			tracePhoton(orig, dir, E, 0u, buffer, tmpHitList);
		}
	}
}

void Renderer::tracePhoton(const FW::Vec3f& orig, const FW::Vec3f& d, const FW::Vec3f& E, const size_t bounce, Node** buffer, std::vector<Hit>& tmpHitList, const float n1)
{
	if(bounce >= maxBounces)
		return;
	
	FW::Vec3f dir = d.normalized();
	Hit hit = Hit(10.f);
	if(!RayTracer::get().rayCast(orig, dir, hit, m_triangles, m_indexListFromScene, m_sceneTree, buffer))
	{
		return;
	}

	const Triangle& tri = hit.triangle;
	FW::MeshBase::Material mat;
	MaterialPM matType = shader(hit, m_mesh, mat);
	FW::Vec3f n = interpolateAttribute(tri, hit.intersectionPoint, m_mesh, m_mesh->findAttrib(FW::MeshBase::AttribType_Normal));
	n = n.normalized();
	FW::Vec3f newOrig = hit.intersectionPoint + .0001f * n;

	if(matType == MaterialPM_Mirror)
	{
		FW::Vec3f newDir = dir - 2.f*FW::dot(dir, n)*n;
		newDir = dir.normalized();
		tracePhoton(newOrig, newDir, E, bounce + 1, buffer, tmpHitList);		
	}
	else if(matType == MaterialPM_GlassSolid)
	{
		const float n2 = mat.opticalDensity;
		const float nDiv = (n1/n2);
		bool toDenser = (n1 <= n2);

		// Dot product of I-ray and surface normal, this is also the cosine of I-ray's angle since Iray and n are unit lenght
		const float dotI = FW::dot(-dir, n);
		
		// If we are moving to lighter, this means we are inside a mesh => flip normal
		if(!toDenser)
			n = -n;

		FW::Vec3f transmittanceDir = (-nDiv * (-dir - dotI*n) -n*FW::sqrt(1.f-(nDiv*nDiv)*(1-(dotI*dotI)))).normalized();
		FW::Vec3f reflectanceDir = (2.f*FW::dot(-dir, n)*n-dir).normalized();

		//Schlick's approximation
		float RShclick;
		float R0 = std::pow(((n1-n2)/(n1+n2)), 2);
		
		//From lighter to denser 
		if(toDenser)
			RShclick = R0 + (1-R0)*std::pow((1.f-dotI),5);
		else
		{
			//Define if there is Total Iternal Reflection
			if(FW::asin(n2/n1) >= FW::acos(dotI))
				RShclick = 1.f;
			else
			{
				//Cosine of T-ray and normal. NB! We are still inside the mesh, but we fliped normal earlier 
				const float dotT = FW::dot(transmittanceDir, -n);
				RShclick = R0 + (1-R0)*std::pow((1.f-dotT),5);
			}
		}

		float r = m_randomGen.getF32(.0f, 1.f);
		if(r < RShclick && toDenser) //To denser, reflected
			tracePhoton(newOrig, reflectanceDir, E * mat.specular, bounce + 1, buffer, tmpHitList);
		else if (r >= RShclick && toDenser) //To denser, refraction
			tracePhoton(newOrig, transmittanceDir, E, bounce + 1, buffer, tmpHitList, n2);
		else if(r < RShclick && !toDenser) //To lighter, reflected
			tracePhoton(newOrig, reflectanceDir, E, bounce + 1, buffer, tmpHitList, n2);
		else if (r >= RShclick && !toDenser) //To lighter, refraction
			tracePhoton(newOrig, transmittanceDir, E, bounce + 1, buffer, tmpHitList);
	}
	else if(matType == MaterialPM_Lightsource)
	{
		FW::Vec3f newDir = randomVectorToHalfUnitSphere(n, m_randomGen);
		tracePhoton(newOrig, newDir, E, bounce+1, buffer, tmpHitList); 
	}
	else if(matType == MaterialPM_Diffuse)
	{
		FW::Vec3f albedo = getAlbedo((TriangleToMeshData*) hit.triangle.m_userPointer, m_mesh, FW::Vec3f((1.0f - hit.u - hit.v, hit.u, hit.v)));
		FW::Vec3f newE = E * FW::dot(n, -dir);
		FW::Vec3f newDir = randomVectorToHalfUnitSphere(n, m_randomGen);

		Photon photon = Photon(hit.intersectionPoint, newE, -(dir.normalized()));
		m_photons.push_back(photon);
		tmpHitList.push_back(hit);

		float r3 = m_randomGen.getF32(0,1.f);
		float threshold = (albedo.x + albedo.y + albedo.z)/3.f;
		if(r3 < threshold)
			tracePhoton(newOrig, newDir, newE * albedo, bounce+1, buffer, tmpHitList);
	}
}

void Renderer::initPhotonMaping(const size_t numOfPhotons, const float r, const size_t FG, const float totalLight, const size_t numberOfSamplesByDimension,const FW::Vec2i& size)
{
	std::cout << "Initiliaze photon maping: " << std::endl;
	std::cout << "Radius - " << r << " / FG rays - " << FG << " / totalLight - " << totalLight << std::endl;
	std::cout << std::endl;

	FW::Timer timerTotal;
	timerTotal.start();
	updateContext(r, FG, totalLight, numberOfSamplesByDimension);
	
	m_photons.clear();
	m_photonIndexList.clear();
	m_photonTestMesh->clear();
	m_photonTestMesh->addSubmesh();
	
	if(m_hasPhotonMap)
		RayTracer::get().demolishTree(m_photonTree);
	
	std::cout << "Starting photon cast...";
	std::vector<Hit> tmpHitList;
	tmpHitList.reserve(numOfPhotons);
	updatePhotonListCapasity(numOfPhotons);
	castPhotons(numOfPhotons, tmpHitList);
	m_hasPhotonMap = true;
	
	std::cout << " done! " << m_photons.size() << " photons total!" << std::endl;
	m_photonTree = RayTracer::get().constructHierarchy(m_photons, m_photonIndexList);

	FW::Timer timer;
	timer.start();
	std::cout << "Precalculate outgoinging light for each photon... " << std::endl;
	preCalculateOutgoingLight(tmpHitList);
	std::cout << " done!  Time spend: " << timer.getElapsed() << std::endl;

	std::cout << "Start image synthesis, image size as pixels: " << size.x << "/" << size.y << "..." << std::endl;
	timer.start();
	synthesisImage(size);
	std::cout << " done!  Time spend: " << timer.getElapsed() << std::endl;
	std::cout << "Everything done... Time spend: " << timerTotal.getElapsed() << std::endl;
	std::cout << "___________________________________________________________________" << std::endl;
	std::cout << std::endl;
	m_renderWithPhotonMaping = true;
}

void Renderer::initImageSynthesisFromExistingPM(const float r, const size_t FG, const float totalLight, const size_t numberOfSamplesByDimension,const FW::Vec2i& size)
{
	if(!m_hasPhotonMap)
	{
		std::cout << "Couldn't start image synthesis, because PM doesn't exist!!!" << std::endl;
		std::cout << "Press 1 for photong casting and image synthesis" << std::endl;
		std::cout << "___________________________________________________________________" << std::endl;
		return;
	}
	FW::Timer timer;
	updateContext(r, FG, totalLight, numberOfSamplesByDimension);
	std::cout << "Start image synthesis based on existing PM, image size as pixels: " << size.x << "/" << size.y << "..." << std::endl;
	std::cout << "Radius - " << r << " / FG rays - " << FG << " / totalLight - " << totalLight << std::endl;
	std::cout << "Attention!!! Changes in amount of totalLight effect direct LS hits, no PM energies!!!" << std::endl;
	timer.start();
	synthesisImage(size);
	std::cout << " done!  Time spend: " << timer.getElapsed() << std::endl;
	std::cout << "___________________________________________________________________" << std::endl;
}

void Renderer::preCalculateOutgoingLight(std::vector<Hit>& tmpHitList)
{
	m_contextData.d_tmpHitList = &tmpHitList;
	m_contextData.d_photonTree = m_photonTree;
	
	m_launcher->popAll();
	m_launcher->setNumThreads(m_launcher->getNumCores());
	//m_launcher->setNumThreads(1);
	m_launcher->push(outgoingLightFunc, &m_contextData, 0, 64);
	while(m_launcher->getNumFinished() != m_launcher->getNumTasks())
	{
		printf("~ %.2f %% \r", 100.0f*m_launcher->getNumFinished()/(float)m_launcher->getNumTasks());
	}
}

void Renderer::outgoingLightFunc(FW::MulticoreLauncher::Task& t)
{
	contextData& data = *(contextData*)t.data;	
	int thread = t.idx;
	int index = thread;
	Node* buffer[1028]; 
	while(index < (*data.d_photons).size())
	{
		Photon& photon = (*data.d_photons)[index];
		Hit& h = (*data.d_tmpHitList)[index];
		std::vector<HeapNode> nodes;
		float r = data.d_FGRadius;
		RayTracer::get().searchPhotons(h.intersectionPoint, *data.d_photons, *data.d_photonIndexList, data.d_photonTree, r, 50u, nodes, buffer);

		if(nodes.empty())
		{
			index += 64;
			continue;
		}

		const Triangle& tri = h.triangle;
		FW::Vec3f normal = (interpolateAttribute(tri, h.intersectionPoint, data.d_mesh, data.d_mesh->findAttrib(FW::MeshBase::AttribType_Normal)));
		normal = normal.normalized();

		const float alpha = 10.818f;
		const float beta = 1.953f;
		const float e = 2.718281f;

		FW::Vec3f Li = FW::Vec3f();
		for ( int j = 1; j < nodes.size(); ++j )
		{		
			FW::Vec3f lightDir = (*data.d_photons)[nodes[j].value].dir;
			float dot = FW::dot(lightDir, normal);
			if(dot < .0f)
				continue;
			float d = nodes[j].key;
			float t1 = 1.0f - pow( e, ( -beta * d * d / ( 2.0f * r * r) ) );
			float t2 = 1.0f - pow( e, -beta );
			float w = alpha * ( 1.0f - (t1 / t2) );
			const Hit& photonHit = (*data.d_tmpHitList)[nodes[j].value];
			TriangleToMeshData* map = (TriangleToMeshData*) photonHit.triangle.m_userPointer;
			FW::Vec3f barys = FW::Vec3f((1.0f - photonHit.u - photonHit.v, photonHit.u, photonHit.v));
			Li +=  w * (*data.d_photons)[nodes[j].value].power * dot * getAlbedo(map, data.d_mesh, barys);
		}
		float A = FW_PI*r*r;
		photon.E = Li / A;
		index += 64;
	}
}

void Renderer::synthesisImage(const FW::Vec2i& size)
{
	if(m_renderWithPhotonMaping)
		delete m_image;

	m_image = new FW::Image(size, FW::ImageFormat::RGBA_Vec4f);
	FW::Vec2i imageSize = m_image->getSize();

	FW::Mat4f worldToCamera = m_camera->getWorldToCamera();
	FW::Mat4f projection = FW::Mat4f::fitToView(FW::Vec2f(-1,-1), FW::Vec2f(2,2), imageSize)*m_camera->getCameraToClip();
	FW::Mat4f invP = (projection * worldToCamera).inverted();
	
	m_contextData.d_invP = invP;
	m_contextData.d_image = m_image;	

	//m_launcher->setNumThreads(1);
	m_launcher->popAll();
	m_launcher->push(imageScanline, &m_contextData, 0, imageSize.y );
	while(m_launcher->getNumFinished() != m_launcher->getNumTasks())
	{
		printf("~ %.2f %% \r", 100.0f*m_launcher->getNumFinished()/(float)m_launcher->getNumTasks());
	}
}

void Renderer::imageScanline(FW::MulticoreLauncher::Task& t)
{
	contextData& data = *(contextData*)t.data;

	const int y = t.idx;	
	const FW::Vec2f imageSize = data.d_image->getSize();
	Node* buffer[1028];

	for(int x = 0; x < imageSize.x; ++x)
	{	
		const float yP = (y + .5f) / imageSize.y * -2.0f + 1.0f;
		const float xP = (x + .5f) / imageSize.x *  2.0f - 1.0f;
		FW::Vec3f E = FW::Vec3f();
		float totalW = .0f;
		MultiJitteredSamplingWithTentFilter sampling = MultiJitteredSamplingWithTentFilter(data.d_numberOfSamplesByDimension, FW::Vec2f(xP, yP), imageSize);
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
			E += traceRay(Ro, Rd, data, buffer, 0u);
			totalW += w;
		}
		E *= 1.f/totalW;
		data.d_image->setVec4f(FW::Vec2i(x,y), FW::Vec4f(E, 1.f));
	}
}

FW::Vec3f Renderer::finalGathering(const FW::Vec3f& pos, const FW::Vec3f& normal, const contextData& data, Node** buffer, const size_t rays)
{
	if(rays == 1)
	{
		float r = data.d_FGRadius;
		int index = RayTracer::get().findNearestPhoton(pos, *data.d_photons, *data.d_photonIndexList, data.d_photonTree, r, buffer);
		if(index == -1)
			return FW::Vec3f();
		else
			return (*data.d_photons)[index].E;
	}
	else
	{
		FW::Vec3f org = pos + 0.001f * normal;
		FW::Vec3f total = FW::Vec3f();
		FW::Random random = FW::Random();
		for(auto i = 0u; i < data.d_numberOfFGRays; ++i)
		{	
			FW::Vec3f dir = randomVectorToHalfUnitSphere(normal, random);
			Hit hit = Hit(10.f);
			total += traceRay(org, dir, data, buffer, 0u, true);
		}
		total *= 1.f/(float) rays;
		return total;
	}
};

FW::Vec3f Renderer::traceRay(const FW::Vec3f& orig, const FW::Vec3f& d, const contextData& data, Node** buffer, const size_t bounce, bool FGRay, const float n1)
{
	const FW::Vec3f dir = d.normalized();
	MeshC* mesh = data.d_mesh;

	if(bounce >= maxBounces)
		return FW::Vec3f(.95, .0f, .95f);
	
	Hit hit = Hit(10.f);
	if(!RayTracer::get().rayCast(orig, dir, hit, *(data.d_triangles), *(data.d_indexListFromScene), data.d_sceneTree, buffer))
	{
		hit.t = -1.f;
		return FW::Vec3f();
	}

	const Triangle& tri = hit.triangle;
	FW::MeshBase::Material mat;
	MaterialPM matType = shader(hit, mesh, mat);
	FW::Vec3f n = (interpolateAttribute(tri, hit.intersectionPoint, mesh, mesh->findAttrib(FW::MeshBase::AttribType_Normal)));
	n = n.normalized();
	FW::Vec3f newOrig = hit.intersectionPoint + 0.0001f * n;

	if(matType == MaterialPM_Lightsource)
	{
		Triangle& tri = hit.triangle;
		FW::Vec3f albedo = getAlbedo((TriangleToMeshData*) hit.triangle.m_userPointer, mesh,FW::Vec3f(1.f-hit.u-hit.v, hit.u, hit.v));
		float dot = FW::dot(-dir, n);	
		if(dot < 0)
			dot = FW::dot(-dir, -n);
				
		return *(*data.d_triangles)[hit.i].m_lightPower * albedo * dot * data.d_totalLight; 
	}
	else if(matType == MaterialPM_Diffuse)
	{
		FW::Vec3f albedo = getAlbedo((TriangleToMeshData*) hit.triangle.m_userPointer, mesh, FW::Vec3f(1.f-hit.u-hit.v, hit.u, hit.v));
		size_t rays = 1u;
		if(!FGRay)
			rays = data.d_numberOfFGRays;
		
		return albedo * finalGathering(newOrig, n, data, buffer, rays);
	}
	else if(matType == MaterialPM_Mirror)
	{
		FW::Vec3f newDir = dir - 2.f*FW::dot(dir, n)*n;
		newDir = newDir.normalized();
		return traceRay(newOrig, newDir, data, buffer, bounce + 1, FGRay);		
	}

	/*Microfaced BRDF*/
	float n2;
	if(n1 < 1.001f)
		n2 = mat.opticalDensity;
	else
		n2 = 1.f;

	const float nDiv = (n1/n2);
	bool toDenser = (n1 < n2);
	if(!toDenser)
		n *= -1.f;

	FW::Vec3f reflectionDir = (2.f*FW::dot(-dir, n)*n-dir).normalized();
	
	// Dot product of I-ray and surface normal, this is also the cosine of I-ray's angle since Iray and n are unit lenght
	float dotI = FW::dot(-dir, n);
	if(dotI < 0.f)
	{
		return traceRay(hit.intersectionPoint + n*.0001f, reflectionDir, data, buffer, bounce + 1, FGRay);
	}

	//Fresnel term -> Schlick's approximation
	float RShclick = 0.f;
	FW::Vec3f refractionDir = FW::Vec3f();
	bool TIR = false;

	if(!toDenser)
	{
		float critalangle = std::asin(n2/n1);
		float incomingAngle = std::acos(dotI);
		if(critalangle <= incomingAngle)
		{
			TIR = true;
			RShclick = 1.f;
		}
	}

	if(!TIR)
		refractionDir =  (nDiv*dir-(nDiv*dotI+FW::sqrt(1.f-(nDiv*nDiv)*(1.f-dotI*dotI)))*n).normalized();

	float R0 = std::pow(((n1-n2)/(n1+n2)), 2);


	if(toDenser)
		RShclick = R0 + (1-R0)*std::pow((1.f-dotI),5);
	else
	{
		if(!TIR)
		{
			//Cosine of T-ray and normal. NB! We are still inside the mesh, but we fliped normal earlier
			float dotT = FW::dot(refractionDir, -n);
			RShclick = R0 + (1-R0)*std::pow((1.f-dotT),5);
		}
	}

	/*//Distripution term, Beckmann NDF
	const float e = 2.718281f;
	const float m = mat.roughness;
	const FW::Vec3f hVec = (-dir + (data.d_lightPosEstimate-hit.intersectionPoint).normalized())*.5f;
	const float dotHN = FW::dot(hVec, n);
	const float exp = (dotHN*dotHN-1.f)/(m*m*dotHN*dotHN);
	const float D = std::pow(e,exp)/(m*m*std::pow(dotHN, 4));

	//Calculate BRDF with visibility term being 1.f
	float BRDF = RShclick * D * .25f;*/

	if(matType == MaterialPM_GlassSolid)
	{
		FW::Vec3f reflection = FW::Vec3f();
		FW::Vec3f refraction = FW::Vec3f();
		if(bounce < 5u)
		{
			if(toDenser)
				reflection = RShclick * traceRay(hit.intersectionPoint + n*.0001f, reflectionDir, data, buffer, bounce + 1, FGRay);
			else
				reflection = RShclick * traceRay(hit.intersectionPoint + n*.0001f, reflectionDir, data, buffer, bounce + 1, FGRay, n2);
		}
		if(!TIR)
		{
			float tmp = FW::dot(refractionDir, -n);
			if(toDenser)
				refraction = (1.f-RShclick)*traceRay(hit.intersectionPoint - n*.0001f, refractionDir, data, buffer, bounce + 1, FGRay, n2);
			else
				refraction = (1.f-RShclick)*traceRay(hit.intersectionPoint - n*.0001f, refractionDir, data, buffer, bounce + 1, FGRay);
		}

		return reflection + refraction;
	}

	return FW::Vec3f();
}

MaterialPM Renderer::shader(const Hit& h, MeshC* mesh)
{
	return shader(h, mesh, FW::MeshBase::Material());
}

MaterialPM Renderer::shader(const Hit& h, MeshC* mesh, FW::MeshBase::Material& mat)
{
	const TriangleToMeshData* data = (TriangleToMeshData*) h.triangle.m_userPointer;
	mat = mesh->material(data->submeshIndex);
	if(mat.emissive != FW::Vec3f())
		return  MaterialPM_Lightsource;
	if(mat.illuminationModel == 5u)
		return MaterialPM_Mirror;
	if(mat.illuminationModel == 7u)
		return MaterialPM_GlassSolid;
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

void Renderer::updateContext(const float r, const size_t FG, const float totalLight, const size_t numberOfSamplesByDimension)
{
	/*
	Doesn't update image pointer, hitList for outgoing light, invP camera matrix or root to photon hierarchy tree!!!!
	*/
	m_contextData.d_numberOfFGRays=  FG;
	m_contextData.d_numberOfSamplesByDimension = numberOfSamplesByDimension;
	m_contextData.d_FGRadius=  r;
	m_contextData.d_totalLight = totalLight;
	m_contextData.d_vertices = &m_vertices;
	m_contextData.d_triangles = &m_triangles;
	m_contextData.d_triangleToMeshData = &m_triangleToMeshData;
	m_contextData.d_indexListFromScene = &m_indexListFromScene;
	m_contextData.d_photons = &m_photons;
	m_contextData.d_photonIndexList = &m_photonIndexList;
	m_contextData.d_mesh = m_mesh;
	m_contextData.d_sceneTree = m_sceneTree;
}

void Renderer::updatePhotonListCapasity(const size_t numberOfPhotons)
{
	size_t c = m_photons.capacity();
	if(c < numberOfPhotons)
	{
		size_t s = numberOfPhotons - c;
		m_photonIndexList.reserve(s);
		m_photons.reserve(s);
	}
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

FW::Vec3f Renderer::getAlbedo(const TriangleToMeshData* map, const MeshC* mesh, const FW::Vec3f& barys)
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

FW::Vec3f Renderer::interpolateAttribute(const Triangle& tri, const FW::Vec3f& p, const FW::MeshBase* mesh, int attribidx )
{
	const TriangleToMeshData* map = (const TriangleToMeshData*)tri.m_userPointer;

	FW::Vec3f barys = getBarys(tri, p, mesh);

	FW::Vec3f v[3];
	v[0] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][0], attribidx ).getXYZ();
	v[1] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][1], attribidx ).getXYZ();
	v[2] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][2], attribidx ).getXYZ();

	return barys[0]*v[0] + barys[1]*v[1] + barys[2]*v[2];
}

FW::Vec3f Renderer::getBarys(const Triangle& tri, const FW::Vec3f& p, const FW::MeshBase* mesh)
{
	const TriangleToMeshData* map = (const TriangleToMeshData*)tri.m_userPointer;
	int posIndex = mesh->findAttrib(FW::MeshBase::AttribType_Position);
	FW::Vec3f pos[3];
	pos[0] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][0], posIndex ).getXYZ();
	pos[1] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][1], posIndex ).getXYZ();
	pos[2] = mesh->getVertexAttrib( mesh->indices(map->submeshIndex)[map->vertexIndex][2], posIndex ).getXYZ();
	
	FW::Vec3f f[3];
	f[0] = pos[0] - p;
	f[1] = pos[1] - p;
	f[2] = pos[2] - p;

	float aTotal = FW::cross(pos[0]-pos[1], pos[0] - pos[2]).length();
	float a1 = FW::cross(f[1], f[2]).length() / aTotal;
	float a2 = FW::cross(f[2], f[0]).length() / aTotal;
	float a3 = FW::cross(f[0], f[1]).length() / aTotal;
	return FW::Vec3f(a1, a2, a3);
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