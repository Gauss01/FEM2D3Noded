import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
file = open("heat.dat", "r")
lines = file.readlines()
nOfElemNodes=3; #triangular elements

constants = {}
data = []
formattedData = {}
readingIndex = 0
for line in lines:
  if line.__contains__("="):
    heading = line.split("=")
    constants[heading[0].strip()] = heading[1].strip()
  else:
    if len(line.strip()) > 0:
      data.append(line.strip())
for constant in constants:
  if constant != "TITLE":
    formattedData[constant] = []
    for line in data[readingIndex:readingIndex+int(constants[constant])]:
        formattedData[constant].append(list(filter(None, line.strip().split(" "))))
    readingIndex += int(constants[constant])

ncoord = formattedData["NODE_COORDINATES"]
ncoord =np.array(ncoord)
nOfNodes = int(constants["NODE_COORDINATES"])

connect = formattedData["ELEMENTS"]
connect = np.array(connect)
nOfElements = int(constants["ELEMENTS"])


# Extract element conductivity (isotropic conductivity assumed)
conductivity = connect[:,nOfElemNodes+1]


# Extract element heat source data

heatSource = connect[:,nOfElemNodes+2];



# Extract connectivity table
elementOrder = connect[:,0]


connect = connect[:,0:nOfElemNodes+1].astype(int);

# Extract nodal coordinates
nodeOrder = ncoord[:,0];

ncoord = ncoord[:,1:nOfElements+2].astype(float);


coordinate={'x1':[],'x2':[], 'x3':[], 'y1':[], 'y2':[], 'y3':[]} 
for j in range(1,nOfElemNodes+1):
    A1 = connect[:,j]
    for i in A1:
        coordinate['x'+str(j)].append(ncoord[i-1][0])
        coordinate['y'+str(j)].append(ncoord[i-1][1])       

#calculating the elemental areas
elemArea =[]
for iElem in range(0,nOfElements):
    elemArea.append(((coordinate['x1'][iElem]-coordinate['x3'][iElem])*(coordinate['y2'][iElem]-coordinate['y3'][iElem])-(coordinate['y1'][iElem]-coordinate['y3'][iElem])*(coordinate['x2'][iElem]-coordinate['x3'][iElem]))/2);
    if  elemArea[iElem] < 0:
        print('numbering is in clockwise')
# Read prescribed temperature

nOfNodesDirichlet= int(constants["NODES_WITH_PRESCRIBED_TEMPERATURE"])


dirichletData =formattedData["NODES_WITH_PRESCRIBED_TEMPERATURE"];
dirichletData=np.array(dirichletData);
# Set fixed dof's information arrays

nodesDirichlet = dirichletData[:,0].astype(int);
valueDirichlet = dirichletData[:,1].astype(float);

# Create a vector of lentgh nOfNodes and set 1 if the node has a imposed
# temperature or 0 otherwise

isNodeFixed = np.zeros((nOfNodes,1));
isNodeFixed[nodesDirichlet-1] = np.ones((nOfNodesDirichlet,1));

# Store a list of nodes with no imposed temperature (this is used in the
# elimintation process)
nOfNodesFree = nOfNodes - nOfNodesDirichlet;
nodesFree = np.zeros((nOfNodesFree,1));

counterNodesFree = 0;
for iNode in range(0,nOfNodes):
    if isNodeFixed[iNode]==0:
        counterNodesFree = counterNodesFree + 1;
        nodesFree[counterNodesFree-1] = iNode;

#% Read convection data
nOfEdgesConvection = int(constants["EDGES_WITH_PRESCRIBED_CONVECTION"])
convectionTable=formattedData["EDGES_WITH_PRESCRIBED_CONVECTION"]
convectionTable = np.array(convectionTable);

convectionNodes = convectionTable[:,0:2];
hCoeff = convectionTable[:,2];
Tinfinity = convectionTable[:,3];


# First and second vertexes of all edges

ConvoCoord={'x0':[],'x1':[], 'y0':[], 'y1':[]} 
for j in range(0,2):
    Ac = convectionTable[:,j].astype(float)   
    for i in Ac:
        ConvoCoord['x'+str(j)].append(ncoord[int(i)-1][0])
        ConvoCoord['y'+str(j)].append(ncoord[int(i)-1][1])       

# Compute the length of the edges on a convection boundary

edgeLength = [];
for iEdge in range (0,nOfEdgesConvection):
    edgeLength.append(np.sqrt((ConvoCoord['x1'][iEdge]-ConvoCoord['x0'][iEdge])**2 + (ConvoCoord['y1'][iEdge]-ConvoCoord['y0'][iEdge])**2))

# Solution phase. Assemble stiffness, forcing terms and solve system

# Assemble global heat rate vector (internal heat sources contribution)

RQ =np.zeros([nOfNodes,1]);

rQ=[];
for iElem in range (0,nOfElements):
    rQ = np.array([1,1,1])*float(heatSource[iElem])*float(elemArea[iElem])/3
    globalElementNodes = connect[iElem,1:nOfElemNodes+1];
    rQindex = 0
    for iEl in globalElementNodes:
        RQ[iEl-1][0] = RQ[iEl-1][0]+rQ[rQindex];
        rQindex+=1
    

# Compute global stiffness matrix
K = np.zeros([nOfNodes,nOfNodes]);

for iElem in range(0,nOfElements):
    Bmatx =1/(float(2*elemArea[iElem]))*np.array([[coordinate['y2'][iElem]-coordinate['y3'][iElem],coordinate['y3'][iElem]-coordinate['y1'][iElem], coordinate['y1'][iElem]-coordinate['y2'][iElem]],[coordinate['x3'][iElem]-coordinate['x2'][iElem], coordinate['x1'][iElem]- coordinate['x3'][iElem],  coordinate['x2'][iElem]-coordinate['x1'][iElem]]])

    B=np.matmul(np.transpose(Bmatx),Bmatx)
    ke = np.array(B)*float(conductivity[iElem])*float(elemArea[iElem])

    globalElementNodes = connect[iElem, 1:nOfElemNodes+1];
    
    for i, arr in enumerate(ke):
        for j, _ in enumerate(ke):
            K[globalElementNodes[i]-1][globalElementNodes[j]-1] += ke[i][j]
# Add convection contributions to stiffness and forcing term
Rinft = np.zeros([nOfNodes,1]);


for iEdge in range(0,nOfEdgesConvection):
    # Compute convection contribution to global stiffness matrix

    hTe =np.array([[ 2 , 1] ,[ 1,  2] ])*(float(hCoeff[iEdge])*float(edgeLength[iEdge])/6);

    globalEdgeNodes = convectionNodes[iEdge,0:2].astype(int);
    #% Compute contribution to global convection forcing vector
    rinft = np.array([1, 1])*(float(hCoeff[iEdge])*float(Tinfinity[iEdge])*float(edgeLength[iEdge]))/2;
    for i, arr in enumerate(globalEdgeNodes):
        for j, _ in enumerate(globalEdgeNodes):
            K[globalEdgeNodes[i]-1][globalEdgeNodes[j]-1]=K[globalEdgeNodes[i]-1][globalEdgeNodes[j]-1] + hTe[i][j]
    # Assembly: Add contribution to global convection forcing vector
    
    rQindex = 0
    for iEl in globalEdgeNodes:
        Rinft[iEl-1][0] += rinft[rQindex];
        rQindex+=1
    #% Assembly: Add contribution to global convection forcing vector
    #Rinfty(globalEdgeNodes) = Rinfty(globalEdgeNodes) + rinfty;
#%--------------------------------------------------------------------------
#% Global right hand side of the system
#%--------------------------------------------------------------------------
F = np.add(RQ, Rinft);



#%--------------------------------------------------------------------------
#% Apply the elimination process to account for Dirichlet boundaries
#%--------------------------------------------------------------------------
#% Retain only equations for nodes that are NOT in a Dirichlet boundary
Kstar = []
for nodeFreeRow in nodesFree:
    ktemp = []
    for nodeFreeCol in nodesFree:
        ktemp.append(K[int(nodeFreeRow[0])][int(nodeFreeCol[0])]) 
    Kstar.append(ktemp)    
    

Fstar = [];
for nodeRow in nodesFree:
    Fstar.append(F[int(nodeRow[0])][0])


# Modify the right hand side with prescribed values
for iRow in  range(0, nOfNodesFree):
    nodeRow = int(nodesFree[iRow][0]);
    Krow  = K[nodeRow,nodesDirichlet-1];
    Fstar[iRow] -= np.matmul(Krow,np.transpose(valueDirichlet));
    
Tstar = np.matmul(np.linalg.inv(Kstar), Fstar)

TstarInit = 0
T = valueDirichlet
for freeNode in nodesFree:
    if(freeNode < len(T)):
        T = np.insert(T, int(freeNode), Tstar[TstarInit])
    else:
       T =  np.append(T, Tstar[TstarInit])
    TstarInit+=1

print(Tstar)
# Post-processing phase. 

# Compute reactions (heat flux) at nodes with prescribed temperature
R = np.zeros([nOfNodes,1]);
for iRow in nodesDirichlet:
    Krow  = K[iRow-1,:];
    R[iRow-1] =np.matmul(Krow, F) - F[iRow-1];

# Compute fluxes
# Plot mesh with heat flow vector distribution

centers = {"x":[], "y":[]}
for i in range(len(coordinate["x1"])):
    print(i)
    centers["x"].append((coordinate["x1"][i] + coordinate["x2"][i] + coordinate["x3"][i])/3); 
    centers["y"].append((coordinate["y1"][i] + coordinate["y2"][i] + coordinate["y3"][i])/3);

q = np.zeros([nOfElements, 2])

for iElem in range(0,nOfElements):
     #   Compute element heat flow vector
    Bmatx =1/(float(2*elemArea[iElem]))*np.array([[coordinate['y2'][iElem]-coordinate['y3'][iElem],coordinate['y3'][iElem]-coordinate['y1'][iElem], coordinate['y1'][iElem]-coordinate['y2'][iElem]],[coordinate['x3'][iElem]-coordinate['x2'][iElem], coordinate['x1'][iElem]- coordinate['x3'][iElem],  coordinate['x2'][iElem]-coordinate['x1'][iElem]]])

    q[iElem] = (np.matmul(Bmatx, [T[i-1] for i in connect[iElem][1:]]) * float(conductivity[iElem]) * -1);

# Plot mesh with temperature distribution
#trisurf(connect,coord(:,1),coord(:,2),T,'facecolor','interp','facelight','phong');

# Plot heat flow vectors with background mesh

plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.axis([0,0.4,0,0.6])
plt.quiver(centers["x"], centers["y"],q[:,0],q[:,1]);


coordinateFinal = {"x" : [], "y": []}
formattedConnect = connect[:,1:];

for i, row in enumerate(formattedConnect):
    print(i)
    for j, col in enumerate(row):
        coordinateFinal["x"].append(ncoord[col-1][0])
        coordinateFinal["y"].append(ncoord[col-1][1])
         

plt.plot(coordinateFinal["x"], coordinateFinal["y"],'-o')





