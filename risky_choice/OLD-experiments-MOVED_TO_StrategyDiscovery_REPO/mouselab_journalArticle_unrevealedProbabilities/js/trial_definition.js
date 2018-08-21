//definition of the gambles
//probabilities=[[0.5,0.25,0.25],[0.1,0.1,0.8]];
//payoffs=[ [[0,100,1],[-1000,10,1],[10,1000,1]],[[100,10],[1,10], [20,50]] ];
//nr_trials=probabilities.length;
process_data=new Array();


//Names of the MouseLab cells
tag = "a0^a1`"
 + "b0^b1`"
 + "c0^c1`"
 + "d0^d1";

//Values that are hidden inside the MouseLab cells
txt = "Gamble1^Gamble 2`"
 + "$169^$235`"
 + "$0^$42`"
 + "$1000^$810`"

//For each cell in the table specify whether it is a MouseLab button (1) or not (0)
state = "0^0`"
 + "1^1`"
 + "1^1`"
 + "1^1";

//Text on the outside of the Mouselab cells
box = "^`"
 + "Payoff for Outcome A of Gamble 1^Payoff for Outcome A of Gamble 2`"
 + "Payoff for Outcome B of Gamble 1^Payoff for Outcome B of Gamble 2`"
 + "Payoff for Outcome C of Gamble 1^Payoff for Outcome C of Gamble 2`";

CBCol = "1^1";
CBRow = "0^1^1^1";
W_Col = "200^200";
H_Row = "50^80^80^80";

chkchoice = false;
btnFlg = 1;
btnType = "button";
btntxt = "Gamble 1^Gamble 2";
btnstate = "1^1";
btntag = "gamble1^gamble2";
to_email = "falk.lieder@berkeley.edu";
colFix = false;
rowFix = false;
CBpreset = true;
evtOpen = 0;
evtClose = 0;

chkFrm=true;
warningTxt = "Some questions have not been answered. Please answer all questions before continuing!";
tmTotalSec = 0;
tmStepSec = 0;
tmWidthPx = 0;
tmFill = false;
tmShowTime = false;
tmCurTime = 0;
tmActive = false;
tmDirectStart = false;
tmMinLabel = "min";
tmSecLabel = "sec";
tmLabel = "timer: ";

//The value in the c-th column of the r-th row is the delay (in ms) after which the contents of cell c will be revealed if cell c is hovered over after having opened cell r.
//There is no delay for the very first cell that you hover over.
//Delay: b0 b1 c0 c1 d0 d1
delay = "1000^1000^1000^1000^1000^1000`"
 + "1000^1000^1000^1000^1000^1000`"
 + "1000^1000^1000^1000^1000^1000`"
 + "1000^1000^1000^1000^1000^1000`"
 + "1000^1000^1000^1000^1000^1000`"
 + "1000^1000^1000^1000^1000^1000";

randomOrder = false;
recOpenCells = false;