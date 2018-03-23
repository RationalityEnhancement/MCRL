<?php
if (isset($_GET['subject'])) {$subject=$_GET['subject'];}
 else {$subject="anonymous";}
if (isset($_GET['condnum'])) {$condnum=$_GET['condnum'];}
	else {$condnum=-1;}

//sleep(2)
    
?>

<HTML>
<HEAD>
<script language="JavaScript" src="js/jquery-min.js"></script>
<script language="JavaScript" src="js/jstat.js"></script>
<script language="JavaScript" src="js/underscore.js"></script>
<script language="JavaScript" src="js/mmturkey.js"></script>
<script language="JavaScript" src="js/parameters.js"></script>
<script language="JavaScript" src="js/jquery.lettering.js"></script>
    <script language="JavaScript" src="js/jquery.textillate.js"></script>
<script language="JavaScript" src="js/MouseLabExperiments.js"></script>
<script language="JavaScript" src="plugins/countdown.js"></script>
<script language="JavaScript" src="js/trial_definition.js"></script>
<script language="JavaScript" src="js/mlweb.js"></script>    
<TITLE>TEST of MouseLab Cell</TITLE>
<link rel="stylesheet" href="css/mlweb.css" type="text/css">    
    <link rel="stylesheet" href="css/animate.css" type="text/css">    

</head>

<body align="center">

    
<!--BEGIN TABLE STRUCTURE-->
<SCRIPT language="JavaScript">
//override defaults

$("#MinutesPerBlock").html(minutes_per_block)
    
mlweb_outtype="CSV";
mlweb_fname="mlwebform";


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

//What is this for?
CBord = "0^1^0^1^2^3`"
+ "1^0^0^1^2^3`"
+ "0^1^0^1^3^2`"
+ "1^0^0^1^3^2`"
+ "0^1^0^2^3^1`"
+ "1^0^0^2^3^1`"
+ "0^1^0^3^2^1`"
+ "1^0^0^3^2^1";

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
activeClass = "actTD";
inactiveClass = "inactTD";
boxClass = "boxTD";
cssname = "css/mlweb.css";
nextURL = "difficult.php";
expname = "camera";
randomOrder = false;
recOpenCells = false;
masterCond = 1;

loadMatrices();    
//inside_matrix=txtcont;
//name_matrix=tagcont;
//outside_matrix=boxcont;
//active_matrix=statecont;

  //Counter-Balancing order?
CBord = "0^1^0^1^2^3`"
        + "1^0^0^1^2^3`"
        + "0^1^0^1^3^2`"
        + "1^0^0^1^3^2`"
        + "0^1^0^2^3^1`"
        + "1^0^0^2^3^1`"
        + "0^1^0^3^2^1`"
        + "1^0^0^3^2^1";

var CBorder=ExpMatrix(CBord);
     
$(document).ready(function() {    
    timefunction('onload','body','body')
});                      

    nr_points=0;
    
</SCRIPT>
<!--END TABLE STRUCTURE-->    
    
<FORM name="mlwebform" onSubmit="return checkForm(this)" method="POST" action="save.php">
<INPUT type=hidden name="procdata" value="">
<input type=hidden name="subject" value="">
<input type=hidden name="expname" value="">
<input type=hidden name="nextURL" value="">
<input type=hidden name="choice" value="">
<input type=hidden name="condnum" value="">
<input type=hidden name="to_email" value="">
</FORM>
<!--BEGIN preHTML-->
    
    <H1>Gambling Game</H1>
    
<div id="trial" align="center">    
<table align="center">
<tr>
    <td width='150'>
        <h2>Round #<span id="blockNrDisplay" class="NrDisplay">1</span></h2>
    </td>

    <td width='150'>
        <h2>Choice #<span id="trialNrDisplay" class="NrDisplay">1</span></h2>
    </td>
    <td width='150'>
        <h2><span id="NrPoints" class="NrDisplay">0</span> Points</h2>
    </td>
    <td width='150'>        
        <script language="JavaScript">
    
            function countDownComplete(arg){
                abortBlock();
            }
    
            var myCountdown1 = new Countdown({time:60*minutes_per_block,onComplete: countDownComplete,hideline:true,rangeHi:"minute",style:"boring",width:100});
        </script>
    </td>
</tr>

</table>
    <p class="block-text"> The smallest payoff is <span id="MinPayoff"></span>. The largest payoff is <span id="MaxPayoff"></span></font></b>.</p>
    <p class="block-text">Please click on the gamble you would like to play, or click on "No Thanks!" if you prefer not to gamble.</P>
<!--END preHTML-->

<!-- MOUSELAB TABLE -->

<table>
    <tr><td><div id="MouseLabTable" align="center">empty</div></td><td width="50"></td> <td width="100" valign="top"> <div id="Outcome" style="font-size:24pt;"></div></td></tr></table>
<!-- END MOUSELAB TABLE -->
    
<!--BEGIN postHTML-->
<!--END postHTML-->

<script language="JavaScript">    
    start_block();    
</script>

</div>
    
<div id="finished" style="display:none" class="block-text">
    The time is up. This round is over.<br/> 
    
    <div id="finishedFeedback" class="block-text">
    In this round you made <span id="NrTrialsCompleted">0</span> choices and won <span id="PointsTotal">0</span> points.                
    </div>

    <div id="NextUp">
    <h2>Next Round</h2>
    <div id="feedbackAnnouncement" class="block-text"></div>
    
    You will be given <span id="MinutesPerBlock">2</span> minutes to win as many points as possible. To proceed to the next round please click "Next".
    <button id="nextButton" onclick="start_block()">Next</button>
    </div>
    
    <div id="Debriefing" style="display:none;">
        <h2>Finished!</h2>
        You have completed the last round of the gambling game. You won <span id="totalNrPointsWon">0</span> points in total. Thank you very much for participating in our experiment! We will review your submission shortly. If you submitted quality work, then you will receive a bonus of &#36;<span id="bonus">0</span> for the number of points you have earned.
    </div>
    <!--
    <h1>Data</h1>
    The following process data were collected:<br>
    <div id="dataPrintOut"></div>
    -->
</div>




    
</body>
    
</html>