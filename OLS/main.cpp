/*
 * main.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: mill2171
 */

#include "main.h"

#define VERBOSE

using namespace std;
using namespace arma;

// global objects accessible to everything in the driver program

vector<Market> mkts;					// stores all our markets.
vector<vector<vector<double>>> inst; 	// vector of instruments; moments x instruments x observations
mat weight_matrix;						// weight matrix for calculating moments

vector<bool> constraints;

ofstream logout;			// optimization log file
ofstream mpilog;			// mpi communication and timing log file
bool mpi_enabled = false;	// does this execution of the program use MPI?
int worldsize;				// how many nodes are there for this execution?
vector<int> priority_list;	// which markets should we send for execution first?

////////////////////////////////
// MAIN ENTRY POINT
///////////////////////////////
int main(int argc, char* argv[]) {

	// set up the environment
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// check if MPI is active
	mpi_enabled = (worldsize > 1);

	// if we're not in charge, we want to load the data and immediately go into our processing loop
	if (rank != 0) {
		load_data(false);
		SubMain();
		// we get here after the main node has told us to shut down
		MPI_Finalize();
		return 0;
	}

	cout << "************************" << endl;
	cout << "************************" << endl;
	cout << "MEDICARE ESTIMATOR START" << endl;
	cout << "************************" << endl;
	cout << "************************" << endl;

	cout << "Notes on this run:" << endl;
	cout << "\t39 markets" << endl;
	cout << "\t8 cost parameters across demographic covariates and avg_pop" << endl;
	cout << "\tFull run with additional sanity checks in obj_f and optimize_state" << endl;
	cout << "\tGreatly increased eps in all upper-level numeric derivatives" << endl;
	cout << "\tStatic estimator! NO SWITCHING COSTS!" << endl;
	// start timer and set up log file
	using namespace std::chrono;
	high_resolution_clock::time_point startTime = high_resolution_clock::now();
	time_t now;
	char the_date[100];
	the_date[0] = '\0';
	now = time(NULL);
	struct tm * timeinfo = localtime(&now);
	strftime(the_date, 100, "%F-%T.txt", timeinfo);
	logout.open(the_date);

	cout << "Logfile opened at " << the_date << endl;

	if (mpi_enabled) {
		logout << "MPI mode enabled. Please look to the *mpi.txt for MPI logging." << endl;
		cout << "MPI mode enabled. Please look to the *mpi.txt for MPI logging." << endl;
	} else {
		cout << "Please input a title for the log file: " << endl;
		string logTitle;
		getline(cin, logTitle);
		logout << "USER TITLE: " << logTitle << endl;
	}

	cout.precision(16);
	load_data(true);

	if (mpi_enabled) {
		strftime(the_date, 100, "%F-%T-mpi.txt", timeinfo);
		mpilog.open(the_date);
		mpilog << "MPI Communication log for " << the_date << endl;
		mpilog << "MPI world size: " << worldsize << endl;

		// don't want to run the job if we're going to be idling nodes often
		if ((worldsize - 1) > mkts.size()) {
			mpilog << "ERROR: More nodes, " << worldsize << ", than markets, " << mkts.size() << "." << endl;
			cout << "ERROR: More nodes, " << worldsize << ", than markets, " << mkts.size() << "." << endl;
			abort();
		}
		mpilog << endl << "node\tmarket\tcommand\telapsed\tparameters" << endl;
	}

	logout << "Starting run..." << endl;

	//mkts[0].simulate_longrun(mkts[0].stratMat, 5);


	cout.precision(16);
	Counterfactuals();
	/*
	cout << "inclusive values" <<endl;
	vector<double> ivs(mkts.size());
	for (unsigned int i=0; i < mkts.size(); i++) {
		ivs[i]=mkts[i].InclusiveValueFromData();
		cout << ivs[i] << endl;
		//cout << mkts[i].InclusiveValueFromData() << endl;
	}
	cout << "average: " << (sum_kahan(ivs)/double(mkts.size())) << endl;
	*/
	//FullRun();

	// set up starting values
	//vector<double> testCost(8,0.0);
	// f: [4.426563099185122, 9.590541507759028, 0.1738870975292311, 0.116197136769922, 0.1034853250567106, -0.05265686949066449, 0.05011456132547755]
	//   f: 4.197242565193768, 0.1299737525303133, 9.474131463774384, 0.07725583311687263, -0.0795773422363675, 0.1022307098797671, 0.01560285754481473, 0.04192009770398668}

	/*
	testCost[0] = 4.197242565193768;
	testCost[1] = 0.1299737525303133;
	testCost[2] = 9.474131463774384;
	testCost[3] = 0.07725583311687263;
	testCost[4] = -0.0795773422363675;
	testCost[5] = 0.1022307098797671;
	testCost[6] = 0.01560285754481473;
	testCost[7] = 0.04192009770398668;

	// optimize market
	mkts[0].OptimizeAndCalcMomentComponents(testCost);
	cout << "Policy function!" << endl;
	for (unsigned int i = 0; i < 101; i++) {
		double share = double(i)/1000.0;
		Strategy s= mkts[0].OptimizeState(testCost, share, share,0).strat;
		cout << share << "\t" << s.g0 << "\t" << s.p1 << "\t" << s.g1 << endl;
	}
	*/


	/*
	for (unsigned int i = 0; i < mkts.size(); i++) {
		vector<double> profits = mkts[i].GetAllProfits(testCost);
		for (unsigned int j = 0; j < profits.size(); j++)
			cout << profits[j] << endl;
	}
	*/
	logout.close();

	high_resolution_clock::time_point endTime = high_resolution_clock::now();
	duration<double> elapsed = duration_cast<duration<double>>(endTime-startTime);
	cout << "Total elapsed time: " << elapsed.count() << " seconds." << endl;

	if (mpi_enabled) {
		// have to tell all the sub processes that we're shutting down
		for (int i = 1; i < worldsize; i++) {
			//world.isend(i, message_tags::msg_finished, 0);
			int mkt = 0;
			MPI_Send(&mkt, 1, MPI_INTEGER, i, message_tags::msg_finished,MPI_COMM_WORLD);
			mpilog << "Told node " << i << " to shut down." << endl;
		}
		mpilog.close();


	}
	// even if MPI isn't active on this run, we should still finalize the environment
	MPI_Finalize();
	return 0;
}

void Counterfactuals() {
	cout.precision(16);

	// set up starting values
	vector<double> testCost(8,0.0);
	// f: [4.426563099185122, 9.590541507759028, 0.1738870975292311, 0.116197136769922, 0.1034853250567106, -0.05265686949066449, 0.05011456132547755]
	//   f: 4.197242565193768, 0.1299737525303133, 9.474131463774384, 0.07725583311687263, -0.0795773422363675, 0.1022307098797671, 0.01560285754481473, 0.04192009770398668}

	testCost[0] = 4.197242565193768;
	testCost[1] = 0.1299737525303133;
	testCost[2] = 9.474131463774384;
	testCost[3] = 0.07725583311687263;
	testCost[4] = -0.0795773422363675;
	testCost[5] = 0.1022307098797671;
	testCost[6] = 0.01560285754481473;
	testCost[7] = 0.04192009770398668;
	//cout << "Step 1: model predicted strategy at parameters" << endl;
	mkts[0].OptimizeAndCalcMomentComponents(testCost);
	cout << endl << mkts[0].OptimizeState(testCost, .051506, 0.0463801, 0) << endl;
	//cout << "Inclusive value: " << endl;
	cout << mkts[0].InclusiveValueFromData() << endl;

	cout << "Step 2: model predicted strategy using solver" << endl;
	mkts[0].Solve(testCost);

	OptimalReturn ret =mkts[0].OptimizeState(testCost, 0.0, 0.0, 0);
	cout << endl << "Strategy at 0-0" << endl << ret << endl;

	ret =mkts[0].OptimizeState(testCost, .051506, 0.0463801, 0);
	cout << endl << "Strategy at 0.051-0.046" << endl << ret << endl;
	/*
	cout << "Step 3: Counterfactual: plus one percent" << endl;
	mkts[0].bench[0] *= 1.01;
	mkts[0].bench[1] *= 1.01;
	mkts[0].Solve(testCost);
	ret =mkts[0].OptimizeState(testCost, .051506, 0.0463801, 0);
	cout << endl << ret << endl;
	*/

	cout << "Step 4: Counterfactual: minus five percent" << endl;
	//mkts[0].bench[0] /= 1.01;
	//mkts[0].bench[1] /= 1.01;
	mkts[0].bench[0] *= .95;
	mkts[0].bench[1] *= .95;
	mkts[0].Solve(testCost);

	ret =mkts[0].OptimizeState(testCost, 0.0, 0.0, 0);
	cout << endl << "Strategy at 0-0" << endl << ret << endl;

	ret =mkts[0].OptimizeState(testCost, .051506, 0.0463801, 0);
	cout << endl << "Strategy at 0.051-0.046" << endl << ret << endl;
	//cout << "Inclusive Value" << endl;
	//cout << mkts[0].InclusiveValueFromData() << endl;
}

void StandardErrors() {
	// set up starting values
	vector<double> testCost(14);

	testCost[0] =1.598311027473455;
	testCost[1] =0.2078970399415927;
	testCost[2] =0.04248764406024737;
	testCost[3] =9.228922643599553;
	testCost[4] =0.2269219868284336;
	testCost[5] =1.233740572125861;
	testCost[6] =-1.109483122660648;
	testCost[7] =-0.01322420693574916;
	testCost[8] =-0.02006367921497538;
	testCost[9] =1.170598739408425;
	testCost[10] =-.158338234502942;
	testCost[11] =-4.831154684636733;
	testCost[12] =2.272613581204305;
	testCost[13] =1.909403419876576;


	cout << "Getting instruments from initial guess" << endl;
	CalcInst(testCost);

	cout << "Setting weight matrix to identity..." << endl;
	weight_matrix = mat(NumMoments(), NumMoments(), fill::eye);
	//gsl_matrix_set_identity(weight_matrix);

	cout << "test standard errors" << endl;
	vector<double> se = standard_errors(testCost, weight_matrix);
	cout.precision(16);
	cout << "estimates: " << testCost << endl;
	cout << "std error: " << se << endl;
	logout.precision(16);
	logout << "estimates: " << testCost << endl;
	logout << "std error: " << se << endl;
}

void FullRun() {
	/* parameters are
	base cost
	(const avg_pop) health age
		0		1	2		3
	gen cost
	(const) health age
	   4	5      6
	gen2 cost
	const
	7
	*/

	// set up starting values based on previous optimum
	vector<double> testCost(8,0.0);
	/*
	testCost[0] = 4.4156;
	testCost[1] = 9.1423;
	testCost[2] = .0850;
	testCost[3] = .0494;
	testCost[4] = .0309;
	testCost[5] = .0095;
	testCost[6] = .0450;
	*/
	testCost[0] = 4.0;
	testCost[1] = 0.05;
	testCost[2] = 9.0;
	testCost[7] = 0.04;


	cout.precision(16);
	// set up constraints
	constraints = vector<bool>(testCost.size(), false);
	constraints[7] = true;


	cout << "Getting instruments from initial guess" << endl;
	CalcInst(testCost);
	cout << "Optimizing with identity weight matrix!" << endl << endl;
	weight_matrix = mat(NumMoments(), NumMoments(), fill::eye);

	// set starting point and initial NM simplex step
	vector<double> start = testCost;
	vector<double> nmstep(start.size(), .2);
	nmstep[0] = 1.0;
	nmstep[2] = 1.0;
	nmstep[3] = 0.1;
	nmstep[7] = 0.01;


	// do a pass
	int status = nelder_mead_opt(start, nmstep, 1e-2,false);
	//cout << "did a nelder mead pass and got status " << status << " with vector " << start << endl;
	//int status = bfgs_opt(start, 0.1,1e-2,100);
	//cout << "did a bfgs pass and got status " << status << " with vector " << start << endl;

	vector<double> firstStageResult = start;

	cout << "Calculating new optimal instruments..." << endl;
	CalcInst(firstStageResult);

	cout << "Calculating S matrix from non-weighted calculated best..." << endl;
	mat s = sample_s(firstStageResult);

	cout << "Calculating weight matrix from S matrix..." << endl;
	weight_matrix = inv(s);

	cout << "Re-estimating based on new weight matrix and optimal instruments!" << endl << endl;

	// want to make the starting simplex a little bit smaller
	// and move our point to the middle of the simplex
	for (unsigned int i =0; i < nmstep.size(); i++) {
		nmstep[i] *= 0.5;
		start[i] -= 0.5*nmstep[i];
	}


	status = nelder_mead_opt(start, nmstep, 1e-3,false);
	cout << "did a nelder mead pass and got status " << status << " with vector " << start << endl;

	vector<double> secondStageResults = start;

	cout << "Calling standard errors for second pass results..." << endl;
	vector<double> se = standard_errors(secondStageResults, weight_matrix);
	cout.precision(16);
	cout << "Parameters: " << secondStageResults << endl;
	cout << "Standard Errors: " << se << endl;

}

// primary control loop for subsidiary processes
void SubMain() {
	using namespace std::chrono;
	high_resolution_clock::time_point startTime, endTime;
	double elapsed;
	//cout << "entered SubMain" << endl;
	while (1) {
		// get a command from the host
		// commands always start with the market to process
		int mkt;
		MPI_Status s;
		MPI_Recv(&mkt, 1, MPI_INTEGER, 0, MPI_ANY_TAG,MPI_COMM_WORLD, &s);

		// the tag that came with the message tells us what to do
		message_tags cmd = static_cast<message_tags>(s.MPI_TAG);
		//cout << "got command " << cmd << endl;

		// if we've received the finished command, get the heck outta dodge!
		if (cmd == message_tags::msg_finished)
			break;

		// otherwise there will be a second message with the parameter count
		int parm_count;
		MPI_Recv(&parm_count, 1, MPI_INTEGER, 0, message_tags::msg_command_num_parameters, MPI_COMM_WORLD, &s);

		// finally, we can receive the parameters
		vector<double> p(parm_count);
		MPI_Recv((void *)p.data(), parm_count, MPI_DOUBLE, 0, message_tags::msg_command_params, MPI_COMM_WORLD, &s);

		// start the timer for processing the command
		startTime = high_resolution_clock::now();
		if (cmd == msg_command_moments) {
			// do the processing
			vector<vector<double>> res = mkts[mkt].OptimizeAndCalcMomentComponents(p);


			// send the result back to the main function
			//cout << "trying to send moments back" << endl;

			// to send the moments back to the main process, first we have to tell it how many moments we're sending
			int nMom = res.size();
			MPI_Send(&nMom, 1, MPI_INTEGER, 0, message_tags::msg_data_num_moments, MPI_COMM_WORLD);
			// then we have to tell it how many observations we're sending per moment
			int nObs = res[0].size();
			MPI_Send(&nObs, 1, MPI_INTEGER, 0, message_tags::msg_data_num_obs, MPI_COMM_WORLD);

			// then we can loop through and send all the actual data
			for (int i = 0; i < nMom; i++) {
				MPI_Send(res[i].data(), nObs, MPI_DOUBLE, 0, message_tags::msg_data_moments, MPI_COMM_WORLD);
			}

		} else if (cmd == msg_command_inst) {
			//cout << "Calculating instruments..." << endl;
			// do the processing
			vector<vector<vector<double>>> res = mkts[mkt].CalcOptimalInstruments(p);
			// set the result back to the main function
			//cout << "trying to send instruments back" << endl;
			//world.send(0, message_tags::msg_data_inst, res);

			// first we have to tell it how many moments we're sending data for
			int nMom = res.size();
			MPI_Send(&nMom, 1, MPI_INTEGER, 0, message_tags::msg_data_num_moments, MPI_COMM_WORLD);

			// we also have to report how many observations we'll be sending
			//cout << "sending number of observations back" << endl;
			int nObs = res[0][0].size();
			MPI_Send(&nObs, 1, MPI_INTEGER, 0, message_tags::msg_data_num_obs, MPI_COMM_WORLD);
			// now loop through each moment and tell it how many instruments we're sending
			for (int m = 0; m < nMom; m++) {
				int nInst = res[m].size();
				//cout << "sending number of instruments " << nInst << " for moment " << m << endl;
				MPI_Send(&nInst, 1, MPI_INTEGER, 0, message_tags::msg_data_num_inst, MPI_COMM_WORLD);
				// loop through each instrument and send them all
				for (int i = 0; i < nInst; i++) {
					//cout << "sending instrument " << i << endl;
					MPI_Send(res[m][i].data(), nObs, MPI_DOUBLE, 0, message_tags::msg_data_inst, MPI_COMM_WORLD);
				}
			}
		} // end of instrument command

		// we're done with the command, so figure out how long it took us
		endTime = high_resolution_clock::now();
		elapsed = (duration_cast<duration<double>>(endTime-startTime)).count();

		// and send that number to the main program
		MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, message_tags::msg_info_time, MPI_COMM_WORLD);

	} // end of control loop

}

// sets up all instruments to use in the moment function. does rank check and r^2 check on each instrument in turn.
void CalcInst(const vector<double> & c) {
	cout << "CalcInst called with c=" << c << endl;
	logout << "CalcInst called with c=" << c << endl;
	//vector of all possible instruments for all possible moments
	// moments x instruments x observations
	vector<vector<vector<double>>> all_inst(3);

	// get the optimal instruments from the markets.
	vector<vector<vector<double>>> opt_inst = GetAllOptInst(c);

	// start all_inst with a constant
	vector<double> constant(opt_inst[0][0].size(), 1.0);
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(constant);

	// load in the optimal instruments
	for (unsigned int m = 0; m < all_inst.size(); m++) {
		for (unsigned int i = 0; i < opt_inst[m].size(); i++) {
			all_inst[m].push_back(opt_inst[m][i]);
		}
	}

	int curPos;
	vector<double> tmpInst(constant.size(),0.0);
/*
	// market identifier is also an instrument
	vector<double> tmpInst(constant.size(),0.0);
	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), double(i));
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);
*/
	// as is the benchmark
	// in this version, there's 100% correlation between the two benchmark rates, so no need to add them both as instruments
	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].bench[0]);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);
	/*
	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].bench[1]);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);
	*/

	// add the three market-level parameters
	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].doc_md_count);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);

	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].hosp_count);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);

	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].nurs_facil_count);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);


	// and the size of the market
	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].size[0]);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);
	curPos = 0;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		std::fill(tmpInst.begin() + curPos, tmpInst.begin() + curPos+mkts[i].NumObs(), mkts[i].size[1]);
		curPos += mkts[i].NumObs();
	}
	for (unsigned int i = 0; i < all_inst.size(); i++)
		all_inst[i].push_back(tmpInst);

	cout << "All instruments loaded. Beginning elimination checks..." << endl;
	vector<vector<vector<double>>> final(all_inst.size());
	for (unsigned int m = 0; m < final.size(); m++) {
		cout << "Checking moment " << m << endl;
		// start with the first instrument
		mat hnew(all_inst[m][0]);
		final[m].push_back(all_inst[m][0]);
		for (unsigned int i=1; i < all_inst[m].size(); i++) {
			cout << "\tChecking instrument " << i << endl;
			mat h1(all_inst[m][i]);
			mat htest = join_horiz(hnew, h1);
			if (arma::rank(htest) < htest.n_cols) {
				cout << "\t\tKill for rank." << endl;
				continue;
			}
			// check r^2
			mat b = inv(hnew.t() * hnew) * hnew.t() * h1;
			mat e = h1 - hnew*b;
			mat yd = h1 - mat(h1.n_rows,1,fill::ones)*mean(h1);
			mat num = e.t() * e;
			mat denom = yd.t() * yd;
			double r = 1.0 - num(0,0) / denom(0,0);
			cout << "\t\tr^2=" << r << endl;
			if (r > 0.9995) {
				cout << "\t\tkill for r^2" << endl;
				continue;
			}
			// passed both test. add it to the list!

			hnew = htest;
			final[m].push_back(all_inst[m][i]);
		}
		cout << "Done with moment " << m << ". Ended up with " << final[m].size() << " instruments." << endl;
		logout << "Done with moment " << m << ". Ended up with " << final[m].size() << " instruments." << endl;
		logout << hnew << endl;
	}

	// save the instruments
	inst = final;

}

int NumMoments() {
	int num = 0;
	for (unsigned int m = 0; m < inst.size(); m++) {
		num += inst[m].size();
	}
	return num;
}

// calculates the sample variance of the moments \hat(S) = E[g g']
// but problem: moments may have different sample sizes! what to do?!
// for now, assume the different years are uncorrelated and make it block diagonal
mat sample_s(const vector<double> & c) {
	logout << "sample_s called with cost params " << c << endl;
	// grab the components from the market
	//vector<vector<double>> comp = mkts[0].OptimizeAndCalcMomentComponents(c, true);
	vector<vector<double>> comp = GetAllComponents(c);

	// how many moments do we have?
	int nMom = comp.size();

	// how many observations do we have?
	int totObs = comp[0].size();

	// allocate needed matrices
	mat s(nMom, nMom, fill::zeros);
	mat g(nMom,1);
	mat tmp(nMom, nMom);

	// loop through each observation and add E[g g'] to s
	for (int i = 0; i < totObs; i++) {
		// get the components for this vector
		for (int j = 0; j<nMom; j++) {
			g(j,0) = comp[j][i];
		}
		// calculate g g' and store the result in tmp
		tmp = g * g.t();
		// add tmp to our running total in s
		s += tmp;
	}

	// divide the whole thing through by the number of observations we have
	s = s / double(totObs);

	cout << "Calculated hat(S) matrix: " << s << endl;

	logout << "sample_s ended with calculated hat(S) matrix: " << s << endl;

	return s;
}

// calculates \hat{G} = G_n(\hat{\theta}) = 1/n sum partial g / partial theta'
mat sample_g(const vector<double> & c){
	logout << "sample_g called with cost params " << c << endl;
	// figure out precision
	//const static double cbrteps = cbrt(numeric_limits<double>::epsilon());
	const static double cbrteps = 0.005;

	// grab the initial moments from the object
	vector<vector<double>> base_comp = GetAllComponents(c);

	// how many parameters are we calculating the derivative for?
	unsigned int nParams = c.size();

	// how many moments do we have?
	unsigned int nMom = base_comp.size();

	// how many observations do we have?
	unsigned int totObs = base_comp[0].size();

	// allocate memory to store the pieces of the gradient. We need nParams terms for each observation
	// so it's Parameters x Moments x Observations
	vector<vector<vector<double>>> high_comps(nParams);
	vector<vector<vector<double>>> low_comps(nParams);

	// store our derivative epsilons
	vector<double> h(nParams);

	// loop through each of the parameters and grab the elements of the gradient
	for (unsigned int p = 0; p < nParams; p++) {
		// determine the right h for this parameter
		h[p] = cbrteps * max(abs(c[p]),1.0); //(c[p] > 1.0 ? c[p] : 1.0);
		if (h[p] > abs(c[p])) {
			cout << "WARNING in sample_g: parameter " << p << " has a h[p]=" << h[p] << " but c[p] is only " << c[p] << endl;
		}
		//if (h[p] == 0.0)
			//h[p] = cbrteps;

		volatile double t = c[p] + h[p];
		h[p] = t - c[p];
		cout << "param " << p << "=" << c[p] << " cbrteps=" << cbrteps<< " | h=" << h[p] << endl;

		// grab the high moments
		vector<double> tmp = c;
		tmp[p] += h[p];
		high_comps[p] = GetAllComponents(tmp);

		// grab the low moments
		tmp[p] = c[p] - h[p];
		low_comps[p] = GetAllComponents(tmp);
	}

	// for each moment, loop through the observations, and the parameters, and calculate the appropriate partial derivatives
	mat g(nMom, nParams, fill::zeros);

	for (unsigned int j = 0; j < nMom; j++) {
		for (unsigned int p = 0; p < nParams; p++) {
			vector<double> tmp(totObs);
			for (unsigned int i = 0; i < totObs; i++) {
				// calculate this component of the gradient
				//tmp.push_back(( ((high_comps[p])[j])[i] - (base_comp[j])[i] )/h[p]);
				tmp.push_back(( ((high_comps[p])[j])[i] - ((low_comps[p])[j])[i] )/(2.0*h[p]));
			}
			// figure out the component of the matrix
			g(j,p) = sum_kahan(tmp);
		}
	}

	// scale the g matrix
	g = g / double(totObs);
	//gsl_matrix_scale(g, 1.0/double(totObs));

	cout << "sample g: " << g << endl;
	logout << "sample_g ended with calculated G matrix: " << g << endl;
	return g;
}


// calculates standard errors according to equation 7.3.34 of Hayashi
vector<double> standard_errors(const vector<double> & c, const mat & W) {
	logout << "standard_errors called with costParams " << c << endl << "weight matrix " << W << endl;
	cout << "std errors..." << endl;
	cout << "get components: G and S" << endl;
	mat G = sample_g(c);
	mat S = sample_s(c);

	cout << "starting multiplication with G=" << G << endl << "W=" << W << endl << "S=" << S << endl;

	//int k = G.n_rows;	// number of moments (never actually used any more)
	int p = G.n_cols;	// number of parameters

	mat invGWG = inv(G.t()*W*G);
	mat tmp_pp = invGWG * G.t() * W * S * W * G * invGWG;
	// tmp_pp now contains our asymptotic variance
	cout << "asymp variance: " << tmp_pp;

	vector<double> stderr(p);
	for (int i=0; i<p; i++) stderr[i] = sqrt(tmp_pp(i,i));

	cout << "got standard errors " << stderr << endl;
	logout << "standard_errors finished with " << stderr << endl;

	return stderr;
}


int nelder_mead_opt(vector<double> & g, const vector<double> & s, double tol, bool rand, int maxiter) {
	cout << "Starting Nelder-Mead pass" << endl;
	logout << "Starting Nelder-Mead pass at " << g << endl;

	// convert guess and step to a gsl_vector
	gsl_vector * guess = gsl_vector_create_from_std(g);
	gsl_vector * step = 0;
	if (s.size() > 0)
		step = gsl_vector_create_from_std(s);

	// allocate nelder mead
	gsl_multimin_fminimizer * min2;
	if (rand)
		min2 = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2rand, g.size());
	else
		min2 = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, g.size());

	// allocate and set function object
	gsl_multimin_function gsl_min_func2;
	gsl_min_func2.n = g.size();
	gsl_min_func2.f=&obj_f;
	gsl_min_func2.params = 0;

	// allocate step if necessary
	gsl_vector * nmstep = gsl_vector_alloc(g.size());
	if (step) {
		gsl_vector_memcpy(nmstep, step);
	} else {
		gsl_vector_set_all(nmstep, 1.0);
	}

	gsl_multimin_fminimizer_set(min2, &gsl_min_func2, guess, nmstep);


	// run iterations of NM optimizer
	int iter = 0;
	int status;
	do {
		iter++;
		status = gsl_multimin_fminimizer_iterate(min2);
		if (status)
			break;

		status = gsl_multimin_test_size(min2->size, tol);

		// output the status
		cout << endl << "primary loop iter " << iter << " | x ";
		cout << (gsl_multimin_fminimizer_x(min2));
		cout << " | size " << min2->size;
		cout << " | f " << gsl_multimin_fminimizer_minimum(min2) << endl << endl;

		logout << endl << "primary loop iter " << iter << " | x ";
		logout << (gsl_multimin_fminimizer_x(min2));
		logout << " | size " << min2->size;
		logout << " | f " << gsl_multimin_fminimizer_minimum(min2) << endl << endl;

	} while (status == GSL_CONTINUE && iter < maxiter);

	cout << "finished with status " << status << endl;

	cout << "Current opt parameters:" << gsl_multimin_fminimizer_x(min2);
	cout << endl << "moment was " << gsl_multimin_fminimizer_minimum(min2) << endl;

	logout << "Nelder Mead pass ended with status " << status << endl;
	logout << "min->x is " << min2->x;
	logout<< " f(x) was " << gsl_multimin_fminimizer_minimum(min2) << endl;

	// assign our result to our start
	gsl_vector_memcpy(guess, gsl_multimin_fminimizer_x(min2));
	g = gsl_vector_convert(guess);

	// free memory
	gsl_vector_free(nmstep);
	gsl_multimin_fminimizer_free(min2);
	gsl_vector_free(guess);
	if (step)
		gsl_vector_free(step);

	return status;

}

int bfgs_opt(vector<double> &g, double step, double tol, int maxiter) {
	logout << "Starting BFGS pass at " << g << endl;

	// convert to guess
	gsl_vector * guess = gsl_vector_create_from_std(g);
	// allocate minimizer
	gsl_multimin_fdfminimizer * min = gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_conjugate_fr,g.size());

	// set up minimizer parameters
	gsl_multimin_function_fdf gsl_min_func;
	gsl_min_func.n = g.size();
	gsl_min_func.f = &obj_f;
	gsl_min_func.df = &obj_df;
	gsl_min_func.fdf = &obj_fdf;
	gsl_min_func.params = 0;

	cout << "Starting BFGS pass..." << endl;
	gsl_multimin_fdfminimizer_set(min, &gsl_min_func, guess, step, 0.05);
	int iter = 0;
	int status;

	do {
		iter++;
		status = gsl_multimin_fdfminimizer_iterate(min);
		if (status)
			break;

		status = gsl_multimin_test_gradient(min->gradient, tol);

		cout << endl << "primary loop iter " << iter << " | x ";
		logout << endl << "primary loop iter " << iter << " | f " << min->f << endl;
		cout << min->x;
		cout << " | g ";
		cout << (min->gradient);
		cout << " | f " << min->f << endl << endl;

	} while (status == GSL_CONTINUE && iter < maxiter);


	cout << "finished with status " << status << endl;

	cout << "Current opt parameters";
	cout << (min->x);
	cout << endl << "moment was " << gsl_multimin_fdfminimizer_minimum(min) << endl;

	logout << "ended BFGS pass with status " << status << endl;
	logout << "min->x is " << min->x << endl;
	logout<< "got moment " << gsl_multimin_fdfminimizer_minimum(min) << endl;

	// copy result
	gsl_vector_memcpy(guess, min->x);
	g = gsl_vector_convert(guess);

	// free memory
	gsl_multimin_fdfminimizer_free(min);
	gsl_vector_free(guess);
	return status;

}


void SendParameterCommand(vector<double> c, message_tags type, int mkt, int dest) {
	//cout << "sending command " << type << " to node " << dest << endl;
	// send the type of command to the node
	MPI_Send(&mkt, 1, MPI_INTEGER, dest, type, MPI_COMM_WORLD);

	// tell the node how many parameters to expect
	int parm_count = c.size();
	MPI_Send(&parm_count, 1, MPI_INTEGER, dest, message_tags::msg_command_num_parameters, MPI_COMM_WORLD);

	// send the parameters themselves
	MPI_Send(c.data(), parm_count, MPI_DOUBLE, dest, message_tags::msg_command_params, MPI_COMM_WORLD);
	//cout << "command sent successfully." << endl;
}



vector<vector<double>> GetAllComponents(const vector<double> & c) {
	// allocate space for the components coming from all of the markets
	vector<vector<vector<double>>> mktComp(mkts.size());
	unsigned int totObs = 0;

if (mpi_enabled) {
	// start the timer
	using namespace std::chrono;
	high_resolution_clock::time_point startTime, endTime;
	double totalTime;
	startTime = high_resolution_clock::now();

	//mpilog << "GetAllComponents called with " << c << endl;
	// set up an array
	vector<int> node_status(worldsize, -1);
	// the home node is never available
	node_status[0] = -2;

	// set up an array telling us which markets need doing
	vector<int> mkt_status(mkts.size(), -1);

	// set up a vector for us to keep track of what comes back first
	typedef pair<const double, int> mkt_time;
	multimap<double, int> inbound;

	// assign initial markets to nodes
	int cur_pri = 0;
	int parm_count = c.size();
	for (int i = 1; i < worldsize; i++) {
		node_status[i] = cur_pri;
		mkt_status[cur_pri] = i;
		// set the "getallcomponents" message to the node
		SendParameterCommand(c, message_tags::msg_command_moments, priority_list[cur_pri], i);
		cur_pri++;
	}

	// wait until we receive something
	bool done = false;
	MPI_Status s;
	while (!done) {
		// find out how many moments were calculated
		int nMom;
		MPI_Recv(&nMom, 1, MPI_INTEGER, MPI_ANY_SOURCE, message_tags::msg_data_num_moments, MPI_COMM_WORLD, &s);
		// which did this come from?
		int src = s.MPI_SOURCE;
		// get the number of observations from that same source
		int nObs;
		MPI_Recv(&nObs, 1, MPI_INTEGER, src, message_tags::msg_data_num_obs, MPI_COMM_WORLD, &s);

		// set up our receiving vector and receive the data
		vector<vector<double>> rec(nMom);
		for (int i = 0; i < nMom; i++) {
			rec[i].resize(nObs);
			MPI_Recv(rec[i].data(), nObs, MPI_DOUBLE, src, message_tags::msg_data_moments, MPI_COMM_WORLD, &s);
		}
		// receive the time it took
		double elapsed;
		MPI_Recv(&elapsed, 1, MPI_DOUBLE, src, message_tags::msg_info_time, MPI_COMM_WORLD, &s);

		// which market was that node working on?
		int mkt = priority_list[node_status[src]];
		// stick that market into our inbound so we know what order to send them out in next time
		inbound.insert(mkt_time(elapsed, mkt));
		// put the data we got from that node into our array
		mktComp[mkt] = rec;
		totObs += rec[0].size();
		// update our status vectors
		mkt_status[node_status[src]] = 0;
		node_status[src] = -1;

		// report to the log
		mpilog << src << "\t" << mkt << "\tcomponents\t" << elapsed << "\t" << c << endl;

		// we're done if every market is done
		done = true;
		for (unsigned int i = 0; i < mkt_status.size(); i++) {
			done = done & (mkt_status[i] == 0);		// markets are done if their status is zero
		}

		// if we aren't done yet, we need to give this node something to do!
		if (!done) {
			// loop through all of the markets and find something for this node to do if any work is left
			for (unsigned int i = 0; i < mkt_status.size(); i++) {
				if (mkt_status[i] == -1) {	// we haven't handled this one yet
					mkt_status[i] = src;	// tell the market array which node we're on
					node_status[src] = i;	// tell the node array which market we're using
					// give the command to the market
					SendParameterCommand(c,message_tags::msg_command_moments, priority_list[i], src);
					break;
				}
			}
		}

	}
	// stop the timer and report
	endTime = high_resolution_clock::now();
	totalTime = (duration_cast<duration<double>>(endTime-startTime)).count();
	mpilog << "all" << "\t" << "total" << "\tcomponents\t" << totalTime << "\t" << c << endl;

	// set up the priority list for next time
	multimap<double, int>::reverse_iterator rit;
	int cur_priority = 0;
	for (rit=inbound.rbegin(); rit!=inbound.rend(); ++rit) {
		//cout << "Got market " << rit->second << " in " << rit->first << " seconds, so I'm putting it in priority list " << cur_priority << endl;
		priority_list[cur_priority] = rit->second;
		cur_priority++;
	}
} else {	// MPI condition
	// aren't using MPI, so just loop through all of them normally
	for (unsigned int i = 0; i < mkts.size(); i++) {
		vector<vector<double>> temp = mkts[i].OptimizeAndCalcMomentComponents(c);
		mktComp[i] = temp;
		totObs += temp[0].size();
	}
}



	// create the final moment array
	vector<vector<double>> moments(mktComp[0].size());
	// allocate memory
	for (unsigned int i =0; i < moments.size(); i++) {
		moments[i].resize(totObs);
	}
	// copy to the right place
	for (unsigned int mom = 0; mom < moments.size(); mom++) {
		//unsigned int curPos = 0;
		vector<double>::iterator iter = moments[mom].begin();
		for (unsigned int mkt = 0; mkt < mkts.size(); mkt++) {
			//cout << "mkt=" << mkt << " | mom=" << mom << " mktComp=" << mktComp[mkt][mom] << endl;
			iter = copy(mktComp[mkt][mom].begin(), mktComp[mkt][mom].end(), iter);
			//cout << "moments[mom] now=" << moments[mom] << endl;
			//curPos += mktComp[mkt][mom].size();
		}
	}
	// apply the instruments if we have anything in the instrument object
	if (inst.size() > 0) {
		vector<vector<double>> instMoments;
		// for each moment in moments, we'll loop through all of the instruments we have in "inst", multiply them, and then throw them into instMoments
		for (unsigned int m=0; m < moments.size(); m++) {
			// add the base moment to inst Moments
			//instMoments.push_back(moments[m]);
			// loop through the instruments
			for (unsigned int j = 0; j < inst[m].size(); j++) {
				// allocate space for the moment
				vector<double> tmp(inst[m][j].size());
				// loop through the observations and calculate the instrumented moment
				for (unsigned int i = 0; i < inst[m][j].size(); i++) {
					tmp[i] = moments[m][i] * inst[m][j][i];
				}
				// add the instrumented moment to our list of moments
				instMoments.push_back(tmp);
			}
		}
		return instMoments;
	}

	return moments;
}

vector<vector<vector<double>>>  GetAllOptInst(const vector<double> & c) {
	cout << "GetAllInst called with " << c << endl;
	logout << "GetAllInst called with " << c << endl;
	// allocate space for the instruments coming from all of the markets
	vector<vector<vector<vector<double>>>> instComp(mkts.size());	// market x moment x instrument x observation
	unsigned int totObs = 0;
if (mpi_enabled) {
	//mpilog << "GetAllOptInst called with " << c << endl;
	// set up an array telling us who is doing what
	vector<int> node_status(worldsize,-1);
	// the home node is never available
	node_status[0] = -2;

	// set up an array telling us which markets need doing
	vector<int> mkt_status(mkts.size(),-1);		// status of -1 means we haven't handled this market yet

	// assign initial markets to nodes
	int cur_mkt = 0;
	for (int i = 1; i < worldsize; i++) {
		node_status[i] = cur_mkt;
		mkt_status[cur_mkt] = i;
		// send the "GetAllOptInst" message to the node
		SendParameterCommand(c, message_tags::msg_command_inst, cur_mkt, i);
		//mpilog << "\tTold node " << i << " to work on market " << cur_mkt << endl;
		cur_mkt++;
	}

	//mpilog << "\t\tnode_status:" << node_status << " | mkt_status: " << mkt_status << endl;

	// wait until we receive something
	bool done = false;
	MPI_Status s;
	while (!done) {
		// find out how many moments were calculated by something
		int nMom;
		MPI_Recv(&nMom, 1, MPI_INTEGER, MPI_ANY_SOURCE, message_tags::msg_data_num_moments, MPI_COMM_WORLD, &s);
		// where did this come from?
		int src = s.MPI_SOURCE;
		// get the number of observations from the same source
		int nObs;
		MPI_Recv(&nObs, 1, MPI_INTEGER, src, message_tags::msg_data_num_obs, MPI_COMM_WORLD, &s);

		// set up our receiving vector
		vector<vector<vector<double>>> rec(nMom);
		// loop through moments
		for (int m=0; m < nMom; m++) {
			// find out how many instruments we have for this moment
			int nInst;
			MPI_Recv(&nInst, 1, MPI_INTEGER, src, message_tags::msg_data_num_inst, MPI_COMM_WORLD, &s);
			rec[m].resize(nInst);
			// loop through the instruments and get the data
			for (int i=0; i < nInst; i++) {
				rec[m][i].resize(nObs);
				MPI_Recv(rec[m][i].data(), nObs, MPI_DOUBLE, src, message_tags::msg_data_inst, MPI_COMM_WORLD, &s);
			}
		}

		// receive the time it took
		double elapsed;
		MPI_Recv(&elapsed, 1, MPI_DOUBLE, src, message_tags::msg_info_time, MPI_COMM_WORLD, &s);


		// which market was that node working on?
		int mkt = node_status[src];
		// put the data we got from that node into our array
		instComp[mkt] = rec;
		totObs += instComp[mkt][0][0].size();
		// update our status vectors
		node_status[src] = -1;
		mkt_status[mkt] = 0;

		// report to the log
		mpilog << src << "\t" << mkt << "\tinstruments\t" << elapsed << "\t" << c << endl;

		// we're done if every market is done
		done = true;
		for (unsigned int i = 0; i < mkt_status.size(); i++) {
			done = done & (mkt_status[i] == 0);		// markets are done if their status is zero
		}

		// if we aren't done yet, we need to give this node something to do!
		if (!done) {
			// loop through all of the markets and find something for this node to do if any work is left
			for (unsigned int i = 0; i < mkt_status.size(); i++) {
				if (mkt_status[i] == -1) {	// we haven't handled this one yet
					mkt_status[i] = src;	// tell the market array which node we're on
					node_status[src] = i;	// tell the node array which market we're using
					// give the command to the market
					SendParameterCommand(c, message_tags::msg_command_inst,i,src);
					break;
				}
			}
		}

	}

} else {
	// we aren't using MPI, so just loop normally
	for (unsigned int i = 0; i < mkts.size(); i++) {
		instComp[i] = mkts[i].CalcOptimalInstruments(c);
		totObs += instComp[i][0][0].size();
	}
}
	// what did we get?
	//cout << "mkts=" << instComp.size() << endl;
	//cout << "moments=" << instComp[0].size() << endl;
	//cout << "instruments=" << instComp[0][0].size() << endl;
	//cout << "observations=" << instComp[0][0][0].size() << endl;

	// create the final instrument array
	vector<vector<vector<double>>> opt_inst(instComp[0].size());	// moment x instrument x observation
	// allocate memory
	for (unsigned int i = 0; i < opt_inst.size(); i++) {
		// give each moment in opt inst the right number of instruments that hold the right number of observations
		opt_inst[i].resize(instComp[0][i].size(), vector<double>(totObs));
	}

	// copy the pieces into the right place
	for (unsigned int mom = 0; mom < opt_inst.size(); mom++) {
		for (unsigned int inst = 0; inst < opt_inst[mom].size(); inst++) {
			vector<double>::iterator iter = opt_inst[mom][inst].begin();
			for (unsigned int mkt = 0; mkt < instComp.size(); mkt++) {
				iter = copy(instComp[mkt][mom][inst].begin(), instComp[mkt][mom][inst].end(), iter);
			}
		}
	}

	// toss it into the instrument frame
	return opt_inst;
}

double obj_f(const gsl_vector * v, void * params) {
	//const unsigned long num_threads = 1;
	cout << "f: " << v << " --- ";
	for (unsigned int i = 0; i < v->size; i++) {
		if ((gsl_vector_get(v,i) < 0.0) && (constraints[i] == true)) {
			cout << "bad input on parameter " << i << ", returning 1e75" << endl;
			return 1e75;
		}
	}

	vector<double> cost = gsl_vector_convert(v);

	// sanity check
	double min_base_cost = cost[0] + cost[1]*1.37 + cost[3]*12;
	if (min_base_cost > 5.315648) {
		cout << "failed sanity check on healthy. min base cost was " << min_base_cost << ", returning 1e75" << endl;
		return 1e75;
	}

	if (min_base_cost + cost[1] > 14.981) {
		cout << "failed sanity check on unhealthy. min base cost was " << min_base_cost << ", returning 1e75" << endl;
		return 1e75;
	}

	vector<vector<double>> comp = GetAllComponents(cost);

	// sum the components into the actual moments
	vector<double> allMoments(comp.size());
	for (unsigned int i = 0; i < comp.size(); i++) {
		allMoments[i] = sum_kahan(comp[i]) / double(comp[i].size());
	}

	// calculate the objective function. use the stored weight matrix if available
	double total;
	if (weight_matrix.n_elem == 0) {
		vector<double> totalComponents;
		for (unsigned int i =0; i < allMoments.size(); i++) {
			totalComponents.push_back(allMoments[i]*allMoments[i]);
		}
		total = sum_kahan(totalComponents);
	} else {
		mat g(allMoments);
		mat tmp = g.t() * weight_matrix * g;
		total = tmp(0,0);
	}
	cout << "tot=" << total << endl;
	logout << "f called with v=" << v << endl;
	logout << "\tmoments returned: " << allMoments << endl;
	logout << "\ttotal: " << total << endl;
	return total;
}


void obj_df(const gsl_vector * v, void *params, gsl_vector *df) {
	cout << "df called" << endl;
	double mid_val = obj_f(v, params);
	//const static double cbrteps = cbrt(numeric_limits<double>::epsilon());
	const static double cbrteps = 0.005;

	// temp will be our argument to valueWrapper. Copy it from v and alter the parameters one-by-one
	gsl_vector * temp = gsl_vector_alloc(v->size);



	long double lowValue, highValue;
	// go through the argument vector one by one, change the necessary value, and call valueWrapper
	for (unsigned int i = 0; i<v->size; i++) {
		// determine the right h for our problem
		double h = cbrteps * max(gsl_vector_get(v, i),1.0);
		//double h = 0.01;
		if (h == 0.0)
			h = cbrteps;

		// these steps ensure that our h is exactly representable in floating point form
		volatile double t = gsl_vector_get(v,i) + h;
		h = t - gsl_vector_get(v,i);

		// copy the input to our temp vector and add h
		gsl_vector_memcpy(temp, v);
		gsl_vector_set(temp, i, gsl_vector_get(temp,i)+h);
		highValue = obj_f(temp, params);

		//gsl_vector_memcpy(temp,v);
		//gsl_vector_set(temp, i, gsl_vector_get(temp,i)-h);
		//lowValue = obj_f(temp, params);

		double diff = double(highValue - mid_val);

		//cout << "df call param " << i << " | high " << highValue << " | low " << lowValue << endl;
		gsl_vector_set(df, i, diff/(h));
	}
	// free memory
	gsl_vector_free(temp);
	cout << "\t\t\tgrad from last df call: " << df << endl;
	logout << "df called v=" << v << " | grad=" << df << endl;

}
void obj_fdf(const gsl_vector * v, void * params, double * f, gsl_vector * df) {
	cout << "obj_fdf called" << endl;
	*f = obj_f(v, params);
	obj_df(v, params, df);
	cout << "obj_fdf ended" << endl;

}


void load_data(bool verbose) {
	if (verbose)
		cout << "Loading markets..." << endl;
	//vector<Market> mkts;

	ifstream fin;

	// open the markets file
	fin.open("/home/petrina/millerk/git/MedicareEstimator/data/39mkt_markets.tab");

	// read each market and add to the list
	int market = 0;

	while(1) {
		int fips;
		double size[2];
		double bench[2];
		double doc_md_count;
		double hosp_count;
		double nurs_facil_count;
		double avg_inc;
		double avg_pop;
		fin >> fips;
		fin >> size[0];
		fin >> size[1];
		fin >> bench[0];
		fin >> bench[1];
		fin >> avg_inc;
		fin >> avg_pop;
		fin >> doc_md_count;
		fin >> hosp_count;
		fin >> nurs_facil_count;

		// check for end of file
		if (fin.eof())
			break;
		Market m(fips, size[0], size[1], bench[0], bench[1], avg_inc, avg_pop, doc_md_count, hosp_count, nurs_facil_count);
		//cout << "got market " << m << endl;
		if (m.fips == 55009) {
			if (verbose)
				cout << "Adding market " << market <<  endl;
			mkts.push_back(m);
			market++;
		}
	}
	fin.close();

	if(verbose) {
		cout << "Done with markets file. Got a total of " << mkts.size() << " markets. " << endl;
		cout << "Loading individuals..." << endl;
	}

	fin.open("/home/petrina/millerk/git/MedicareEstimator/data/individuals.tab");
	// keep our place
	Market * curM = &(mkts[0]);
	int people = 0;
	while(1) {
		int fips;
		int h;
		int in;
		double alpha;
		double beta_i;
		double beta_ig;
		double weight;
		double age;
		double female_flag;
		double black_flag;
		double hispanic_flag;
		double grad_hisch;
		double some_coll;
		double bach_degree;
		// read from file
		fin >> fips;
		fin >> h;
		fin >> in;
		fin >> alpha;
		fin >> beta_i;
		fin >> beta_ig;
		fin >> weight;
		fin >> age;
		fin >> female_flag;
		fin >> black_flag;
		fin >> hispanic_flag;
		fin >> grad_hisch;
		fin >> some_coll;
		fin >> bach_degree;
		//cout << fips << "\t" << h << "\t" << in << "\t" << alpha << "\t" << beta_i << "\t" << beta_ig << "\t" << weight << "\t" << age << "\t" << female_flag << "\t" << black_flag << "\t" << hispanic_flag << "\t" << grad_hisch << "\t" << some_coll << "\t" << bach_degree << endl;
		// check for end
		if (fin.eof())
			break;
		// adjust h because in stata, h==1 is healthy, the rest of this code has h==0 as healthy
		h = 1-h;
		// create person out of the read data
		Person p(fips, in, h, alpha, beta_i, beta_ig, weight, age, female_flag, black_flag, hispanic_flag, grad_hisch, some_coll, bach_degree);

		// check to see if we're in the right place
		if (curM->fips != p.GetMarket()) {
			// we don't have the right market, so find the right one
			int target = p.GetMarket();
			for (unsigned int i = 0; i < mkts.size(); i++) {
				if (mkts[i].fips == target) {
					curM = &(mkts[i]);
					break;
				}
			}
			// did we find the right one?
			if (curM->fips != target) {
				//cout << "WARNING: Couldn't find fips " << target << " from the individual file in the market file." << endl;
				continue;
			}
		}

		// ok, we're at the right place. add the person.
		//cout << "Adding person " << p << " to market " << p.GetMarket() << endl;
		curM->AddPerson(p);
		people++;
	}

	fin.close();
	if (verbose) {
		cout << "Added a total of " << people << " people." << endl;
		cout << "Weighting..." << endl;
		for (unsigned int i = 0; i < mkts.size(); i++) {
			cout << mkts[i].fips << ": " << mkts[i].profitTerms + mkts[i].shareTerms[0] + mkts[i].shareTerms[1] << endl;
		}
	}
	//cout << "Re-weighting..." << endl;
	for (unsigned int i = 0; i < mkts.size(); i++) {
		mkts[i].ReweightPeople();
	}

	fin.open("/home/petrina/millerk/git/MedicareEstimator/data/firms.tab");
	int firms = 0;
	curM = &(mkts[0]);
	while(1) {
		// load in the firm
		int year;
		int fips;
		//string contract_id;
		double share_healthy, share_unhealthy;
		int plan0_exist, plan1_exist;
		double plan0_gen, plan1_price, plan1_gen;
		fin >> year;
		fin >> fips;
		//fin >> contract_id;
		fin >> share_healthy;
		fin >> share_unhealthy;
		fin >> plan0_exist;
		fin >> plan1_exist;
		fin >> plan0_gen;
		fin >> plan1_price;
		fin >> plan1_gen;
		// check for end
		if (fin.eof())
			break;

		// create firm
		FirmObs firm(year, fips, share_healthy, share_unhealthy, plan0_exist, plan1_exist, plan0_gen, plan1_price, plan1_gen);
		// check to see if we're in the right place
		if (curM->fips != firm.market) {
			// we don't have the right market, so find the right one
			int target = firm.market;
			for (unsigned int i = 0; i < mkts.size(); i++) {
				if (mkts[i].fips == target) {
					curM = &(mkts[i]);
					break;
				}
			}
			// did we find the right one?
			if (curM->fips != target) {
				//cout << "ERROR: FIRM FILE HAD A FIPS I COULDN'T FIND!" << endl;
				//exit(1);
				continue;
			}
		}
		// ok, we're at the right place, add the firm
		curM->AddFirm(firm);
		firms++;
	}
	fin.close();
	if (verbose)
		cout << "Done adding " << firms << " firms. Calculating long run states... " << endl;

	for (unsigned int i = 0; i < mkts.size(); i++) {
		mkts[i].CalcCompEnvFromData();
		if (verbose) {
			cout << "Market " << mkts[i].fips << " compEnv shares: " << mkts[i].env.insideShare[0] << " " << mkts[i].env.insideShare[1] << endl;
			cout << mkts[i].env << endl << endl;
			logout << "Market " << mkts[i].fips << " compEnv shares: " << mkts[i].env.insideShare[0] << " " << mkts[i].env.insideShare[1] << endl;
			logout << mkts[i].env << endl << endl;
		}
	}
	// set up the priority list
	priority_list = vector<int>(mkts.size());
	for (unsigned int i = 0; i < mkts.size(); i++)
		priority_list[i] = i;

	if (verbose) {
		cout << "Done loading markets." << endl;
		logout << "Done loading markets." << endl << endl ;
	}
}
