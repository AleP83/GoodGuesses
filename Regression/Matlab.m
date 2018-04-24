Xaux=importdata('/Users/alessandroperi/GoogleDrive/C/Projects/Esperimenti/Regression/Matrix.txt');

%{

aK=repmat(Grids_.kgrid,Grids_.nBRgrid*Grids_.nxgrid*Grids_.nChigrid,1);
aB = kron(Grids_.BRgrid,ones(Grids_.nkgrid,1));
aB = repmat(aB,Grids_.nxgrid*Grids_.nChigrid,1);
aX = kron(Grids_.xgrid,ones(Grids_.nkgrid*Grids_.nBRgrid,1));
aX = repmat(aX,Grids_.nChigrid,1);
aC = kron(Grids_.Chigrid,ones(Grids_.nkgrid*Grids_.nBRgrid*Grids_.nxgrid,1));
%aK2 = aK.^2;
%aB2 = aB.^2;
%aX2 = aX.^2;
%aC2 = aC.^2;
X = [ones(Grids_.nkgrid*Grids_.nBRgrid*Grids_.nxgrid*Grids_.nChigrid,1),aK,aB,aX,aC];
%}

%Y = reshape(Ri_.Vc,Grids_.nkgrid*Grids_.nBRgrid*Grids_.nxgrid*Grids_.nChigrid,[]);
%b = regress(Y,X)

Y = Xaux(:,1);
X = Xaux(:,2:5);

b = regress(Y,X)