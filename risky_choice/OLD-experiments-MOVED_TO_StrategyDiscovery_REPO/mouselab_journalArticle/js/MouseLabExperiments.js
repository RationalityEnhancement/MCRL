/*
 	MouselabWebExperiments  
                
     this script contains functions to generate online experiments comprising multiple trials with MouseLab elements 								
*/

//Some Initialization Code
var block_nr=1;
var newBlock = false;
var trial_nr;
var trial_nr_total = 1;
var sample_nr;
var process_data=new Array(nr_blocks);
var feedback=new Array();
var decision_problems=new Array(nr_blocks);
var choices=new Array(nr_blocks);
var outcomes = new Array;
var netPay = new Array;
var observations = new Array;
//var total_nr_points=0;
var problem_type=new Array(nr_blocks);
var range_nr_outcomes = [4, 4];
var range_nr_gambles = [7, 7];
var isHighCompensatory = new Array();
isHighCompensatory[0] = shuffle([1,1,1,1,1,0,0,0,0,0]);
isHighCompensatory[1] = shuffle([1,1,1,1,1,0,0,0,0,0]);
var nr_trials = 20;
var question_nr=1;
var nr_questions=8;
correct_answers = [1,1,0,1,1,0,1,1];
var failed_quiz = new Array();
var seconds_left = 0;
var trialTime = new Array;
var RTs = new Array(nr_trials);
var startTime_RT, endTime_RT, startTime_trial, endTime_trial, quizTime, experimentTime
var nr_outcomes = Math.floor(Math.random() * (range_nr_outcomes[1]+1 - range_nr_outcomes[0]) + range_nr_outcomes[0]);
var nr_gambles = Math.floor(Math.random() * (range_nr_gambles[1]+1 - range_nr_gambles[0]) + range_nr_gambles[0]);
for (o=0;o<nr_trials;o++){        
    RTs[o]=fillArray(-1,nr_outcomes*nr_gambles);
}
var isFullyRevealed = Math.round(Math.random());
var tmp_2 = 1;
if (isFullyRevealed==1){
    var nr_questions=6;
    correct_answers = [1,0,1,1,0,1];
    condStr = '_isFullyRevealed';
}
else{
    condStr = '';
}

//
acquisitions=new Array(nr_blocks);
nr_acquisitions=new Array(nr_blocks);
nr_acquisitions_by_problem_type=new Array(nr_blocks);
nr_problem_types=4;
choice_type=new Array(nr_blocks);
$("#MinutsPerBlock").html(minutes_per_block)
$("#NrPoints").html(0)

$("#MaxBonus").html(max_bonus)


function nextQuestion(){
    
    //Did they forget to check the box?
    checked_box=$('#Q'+question_nr+'A1'+condStr).is(':checked') || $('#Q'+question_nr+'A2'+condStr).is(":checked")
    
    if (!checked_box){
        alert('Please check the box next to your preferred answer.')
        return
    }
    
    if (question_nr<nr_questions){
        $("#question"+question_nr+condStr).hide()
        question_nr++;
        $("#question"+question_nr+condStr).show()
    }
    else{
        $("#question"+question_nr+condStr).hide()
        $("#question1"+condStr).show()        
        $("#quiz"+condStr).hide();
        question_nr=1;
        scoreQuiz()
    }
}


function scoreQuiz(){    
    
    if ((isFullyRevealed==0 && $("input[name='Quiz1"+"']:checked").val()==correct_answers[0] && 
        $("input[name='Quiz2"+"']:checked").val()==correct_answers[1] &&
        $("input[name='Quiz3"+"']:checked").val()==correct_answers[2] &&
        $("input[name='Quiz4"+"']:checked").val()==correct_answers[3] &&
        $("input[name='Quiz5"+"']:checked").val()==correct_answers[4] &&
        $("input[name='Quiz6"+"']:checked").val()==correct_answers[5] &&
        $("input[name='Quiz7"+"']:checked").val()==correct_answers[6] &&
        $("input[name='Quiz8"+"']:checked").val()==correct_answers[7]
       ) ||
        (isFullyRevealed==1 && $("input[name='Quiz1_isFullyRevealed"+"']:checked").val()==correct_answers[0] && 
        $("input[name='Quiz2_isFullyRevealed"+"']:checked").val()==correct_answers[1] &&
        $("input[name='Quiz3_isFullyRevealed"+"']:checked").val()==correct_answers[2] &&
        $("input[name='Quiz4_isFullyRevealed"+"']:checked").val()==correct_answers[3] &&
        $("input[name='Quiz5_isFullyRevealed"+"']:checked").val()==correct_answers[4] &&
        $("input[name='Quiz6_isFullyRevealed"+"']:checked").val()==correct_answers[5]
//        $("input[name='Quiz7_isFullyRevealed"+"']:checked").val()==correct_answers[6]
       ))
    {
        $("#Quiz"+condStr).hide()
        $("#PassedQuiz"+condStr).show()        
    }
    else{
        $("#Quiz"+condStr).hide()
        $("#FailedQuiz"+condStr).show()
        failed_quiz.push(true);
    }
}


//Function Definitions
function start_block(){
    $("#payoff1Low").html(payoff_range1[0]);
    $("#payoff1High").html(payoff_range1[1]);
    $("#payoff2Low").html(payoff_range2[0]);
    $("#payoff2High").html(payoff_range2[1]);
    $("#block2_stakes").html(block2_stakes);
    if (block_nr==1){
        payoff_range = payoff_range1;
        $("#payoffHigh").html(payoff_range1[1]);
        $("#block_stakes").html(Block1_stakes);
    }
    else if (block_nr==2){
        payoff_range = payoff_range2;
        $("#payoffHigh").html(payoff_range2[1]);
        $("#block_stakes").html(Block2_stakes);
    }
    payoff_mu = (payoff_range[0]+payoff_range[1])/2;
    payoff_std = 0.3*(payoff_range[1]-payoff_range[0])
    if (block_nr==1){
        payoff_mu1 = payoff_mu;
        payoff_std1 = payoff_std;
    }
    else if (block_nr==2){
        payoff_mu2 = payoff_mu;
        payoff_std2 = payoff_std;
    }
    
    $("#nrTrialsDisplay").html(nr_trials);
    trial_nr=1;
    process_data[block_nr-1]=new Array();
    feedback[block_nr-1]=new Array();
    decision_problems[block_nr-1]=new Array();
    choices[block_nr-1]=new Array();
    problem_type[block_nr-1]=new Array();
    nr_acquisitions_by_problem_type[block_nr-1]=[0,0,0,0];
    nr_acquisitions[block_nr-1]=new Array();
    acquisitions[block_nr-1]=new Array();
    choice_type[block_nr-1]=new Array();
    
    nr_points=0;
    $("#finished").hide()
    $("#trial").show()
    $("#blockNrDisplay").html(block_nr)
    $("#MinutsPerBlock").html(minutes_per_block)
    

    $("#NrPoints").html(0)

    if (trial_nr_total-1 == nr_trials/2){
        newBlock = true;
    }
    start_trial(trial_nr);    
}

function start_trial(trial_nr){
    
    start_RTtrial()
    start_RT()
    if (trial_nr_total<=nr_trials){
        $('#trial').hide();
        setTimeout(function(){start_trial2(trial_nr);},100);
    }
    else{
        $("#trial").hide();
        saveAnswers();
    }
}

function start_trial2(trial_nr){
    
    if (isFullyRevealed==1){
        seconds_left = 0;
    }
    else{
        seconds_left = 30;//30;
    }
    var interval = setInterval(function() {
        //document.getElementById('timer_div').innerHTML = --seconds_left;
        seconds_left--
        $("#Timer").html(["You must wait at least <b>"+seconds_left+"</b> seconds before betting"]);

        if (seconds_left <= 0)
        {
           //document.getElementById('timer_div').innerHTML = "You are Ready!";
            $("#Timer").html([""]);
           clearInterval(interval);
        }
    }, 1000);
    
    if (trial_nr_total-1 == nr_trials/2 && newBlock==false){
        block_nr = 2;
        payoff_range = payoff_range2;
        payoff_mu = (payoff_range[0]+payoff_range[1])/2;
        payoff_std = 0.3*(payoff_range[1]-payoff_range[0])
        $("#trial").hide()
        $('#HalfwayMessage').show();
    }
    else{
        $('#trial').show();
        sample_nr = 0;
        hasGambled = false;

        $("#trialNrDisplay").html(trial_nr_total);

        decision_problem = generateGrid(range_nr_outcomes, range_nr_gambles)

        nr_acquisitions[block_nr-1][trial_nr-1]=0;

        decision_problems[block_nr-1][trial_nr-1]=decision_problem;

        $("#FeatureImg").html("<img src='images/"+img_by_problem_type[decision_problem.type-1]+ "' alt='Shape' style='width:150px;height:150px;'>");

        matrices=generateMatrices(decision_problem.probabilities,decision_problem.payoffs);

        matrices = update_pseudorewards(matrices,decision_problem);

        acquisitions[block_nr-1][trial_nr-1]=new Array();
    }
}


function mouselab_table(name_matrix,outside_matrix,inside_matrix){
    
    tableTop='<div style="display: inline-block; text-align:center"> <TABLE border=0 align="left">'
    tableBottom='</TABLE></div>'
    nr_rows=name_matrix.length;
    nr_columns=name_matrix[0].length;
    
    tableHTML=tableTop;
    tableHTML+=choice_buttons_row(nr_columns-1)+"</tr></table><br/><table border=0 align='left'>";
    
    table_rows=new Array();
    for (r=0; r<nr_rows;r++){
        
        table_cells='';
        table_cells+=mouselab_cell(name_matrix[r][0],outside_matrix[r][0],inside_matrix[r][0],active_matrix[r][0])
        for (c=1;c<nr_columns;c++){
            
            //Format number inside the cell
            if (inside_matrix[r][c]>0){
                inside_html="<b><font color='green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$&nbsp;"+inside_matrix[r][c]+"</font></b>"
            }
            if (inside_matrix[r][c]<0){
                inside_html="<b><font color='red'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$&nbsp;"+inside_matrix[r][c]+"</font></b>"
            }
            if (inside_matrix[r][c]==0){
                inside_html="<font color='black'><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$&nbsp;"+inside_matrix[r][c]+"</font></b>"
            }
            
            table_cells+=mouselab_cell(name_matrix[r][c],outside_matrix[r][c],inside_html,active_matrix[r][c]);
        }
        
        table_rows[r]='<tr>'+table_cells+'</tr>';
        tableHTML+=table_rows[r];
    }            
    
    tableHTML+=tableBottom;
    loaded = true;
    return tableHTML;
}

function mouselab_cell(name,outside,inside,active){
    if(active==1){  
        cellType="act"
    }
    else{        
        cellType='inact';
    }    
    cellHTML='<TD align=center valign=middle><DIV ID="'+name+'_cont" style="position: relative; height:'+cell_height+'px; width:'+cell_width+'px;"><DIV ID="'+name+'_txt" STYLE="position: absolute; left: 0px; top: 0px; height:'+cell_height+'px; width:'+cell_width+'px; clip: rect(0px '+cell_width+'px '+cell_height+'px 0px); z-index: 1;"><TABLE><TD ID="'+name+'_td" align=left valign=center width='+(cell_width-5)+' height='+(cell_height-5)+' class="'+cellType+'TD">'+inside+'</TD></TABLE></DIV><DIV ID="'+name+'_box" STYLE="position: absolute; left: 0px; top: 0px; height:'+cell_height+'px; width:'+cell_width+'px; clip: rect(0px '+cell_width+'px '+cell_height+'px 0px); z-index: 2;"><TABLE><TD ID="'+name+'_tdbox" align=center valign=center width='+(cell_width-5)+' height='+(cell_height-5)+' class="'+cellType+'BoxTD">'+outside+'</TD></TABLE></DIV><DIV ID="'+name+'_img" STYLE="position: absolute; left: 0px; top: 0px; height:'+cell_height+'px; width:'+cell_width+'px; z-index: 5;"><A HREF="javascript:void(0);" NAME="'+name+'" onClick="ShowCont(\''+name+'\',event)" onMouseOut="HideCont(\''+name+'\',event)"><IMG NAME="'+name+'" SRC="transp.gif" border=0 width='+cell_width+' height='+cell_height+'></A></DIV></DIV></TD>'
    
//    HideCont(\''+name+'\',event)
    return cellHTML;    
    
}

function choice_buttons_row(nr_choices){
    
    buttons_row_html="<TR>";
    
    //The first button says "No thanks!"
    buttons_row_html+=mouselab_cell("a0","Balls:","Balls",false);
    
    for (c=0; c<nr_choices; c++){
        
        button_name="btn_"+(c+1);
        choice_name="opt"+num_abc(c+1);
        choice_value="Bet "+(c+1);    
            
        buttons_row_html+=choice_button(button_name,choice_name,choice_value);
    }

    
    buttons_row_html+="</TR>";
    
    return buttons_row_html;
}

function choice_button(button_name,choice_name,choice_value){
    
    chosen_gamble=abc_num(choice_name.slice(-1));
    
    choice_button_html='<TD align=center valign=middle><DIV ID="'+button_name+'_cont" style="position: relative; height:'+cell_height+'px; width:'+cell_width+'px;"><DIV ID="'+button_name+'_txt" STYLE="position: absolute; left: 0px; top: 0px; height:'+cell_height+'px; width:'+cell_width+'px; clip: rect(0px '+cell_width+'px '+cell_height+'px 0px); z-index: 1;"><TABLE><TD ID="'+button_name+'_td" align=center valign=center width='+(cell_width-5)+' height='+(cell_height-5)+' class="choiceButton">'+choice_value+'</TD></TABLE></DIV><DIV ID="'+button_name+'_box" STYLE="position: absolute; left: 0px; top: 0px; height:'+cell_height+'px; width:'+cell_width+'px; clip: rect(0px '+cell_width+'px '+cell_height+'px 0px); z-index: 2;"><TABLE><TD ID="'+button_name+'_tdbox" align=center valign=center width='+(cell_width-5)+' height='+(cell_height-5)+' class="choiceButton">'+choice_value+'</TD></TABLE></DIV><DIV ID="'+button_name+'_img" STYLE="position: absolute; left: 0px; top: 0px; height:'+cell_height+'px; width:'+cell_width+'px; z-index: 5;"><A HREF="javascript:void(0);" NAME=\"'+button_name+'\"  onClick="handleButtonClick(\''+choice_name+'\',\''+choice_value+'\',\''+chosen_gamble+'\')\" onMouseOut="HideCont(\''+button_name+'\',event)"><IMG NAME="'+button_name+'" SRC="transp.gif" border=0 width='+cell_width+' height='+cell_height+'></A></DIV></DIV></TD>'
        
    return choice_button_html;
}

function handleButtonClick(choice_name,choice_value,chosen_gamble){

    if (hasGambled==true){
        return;
    }
    if (seconds_left>0){
        return;
    }
    hasGambled = true;
    filled=true; chkchoice=true;
    choices[block_nr-1][trial_nr-1]=chosen_gamble;
    
    if (nr_acquisitions[block_nr-1][trial_nr-1]==0 && chosen_gamble>0){
        choice_type[block_nr-1][trial_nr-1]=1; //Choice Type: Random
    }
    if (nr_acquisitions[block_nr-1][trial_nr-1]==0 && chosen_gamble==0){
        choice_type[block_nr-1][trial_nr-1]=2; //Choice Type: Disengaged
    }
    if (nr_acquisitions[block_nr-1][trial_nr-1]>0){
        choice_type[block_nr-1][trial_nr-1]=3; //Choice Type: Engaged
    }

    hasChosen(chosen_gamble);
    
    //recChoice('onclick',choice_name,choice_value);
    chkchoice = true;
    timefunction('onclick', choice_name, choice_value);
    if (document.forms[mlweb_fname].choice) {document.forms[mlweb_fname].choice.value = name;}

    
    process_data[block_nr-1][trial_nr-1]=document.mlwebform.children[0].value;
    document.mlwebform.children[0].value='';

//    if (trial_nr_total<nr_trials){
        trial_nr++;
        trial_nr_total++;
        $('#NewGame').show();
//    }
//    else{
//        $("#trial").hide();
//        saveAnswers();
//    }
}


function generateGrid(range_nr_outcomes, range_nr_gambles){
    
    probabilities=new Array(nr_outcomes);
    payoffs=new Array(nr_outcomes);
    revealed = new Array(nr_outcomes);
    reveal_order = new Array(nr_outcomes);
    mus = new Array(nr_gambles);
    sigmas = new Array(nr_gambles);
    PRs = new Array(nr_outcomes)
    for (o=0;o<nr_outcomes;o++){        
        payoffs[o]=new Array(nr_gambles);
        revealed[o] = new Array(nr_gambles);
        reveal_order[o] = new Array(nr_gambles);
        PRs[o] = new Array(nr_gambles)
        for (g=0;g<nr_gambles;g++){
            payoffs[o][g] = parseFloat(Math.round(randn_trunc(payoff_mu,payoff_std,payoff_range)*100)/100).toFixed(2)
            mus[g] = payoff_mu;
            sigmas[g] = payoff_std;
            revealed[o][g] = isFullyRevealed;
            reveal_order[o][g] = 0; 
        }
    }
    // force every probability to be >= 1/100?
    psumsum = 0;
    cont = true;
    while (psumsum != 1 || cont){
        cont = true;
        psumsum = 0;
        psum = 0;
        for (o=0;o<nr_outcomes;o++){
            prob = 0;
            while (prob==0){
                prob = Math.round(Math.random()*100)/100;
            }
            probabilities[o]=prob;
            psum+=probabilities[o];
        }
        psumsum = 0;
        for (o=0;o<nr_outcomes;o++){
            if (probabilities[o]<0.01){
                cont = true;
            }
        }
        for (o=0;o<nr_outcomes;o++){   
            probabilities[o] = Math.round(probabilities[o]/psum*100)/100;
            psumsum+=probabilities[o];
        }
        if (isHighCompensatory[block_nr-1][trial_nr-1]){
            for (o=0;o<nr_outcomes;o++){
                if (probabilities[o]>=0.85){
                    cont = false
                }
            }
        }
        else{
            cont = false
            for (o=0;o<nr_outcomes;o++){
                if (probabilities[o]>=0.4 || probabilities[o]<=0.1){
                    cont = true
                }
            }
        }
    }
    
    decision_problem={
    probabilities: probabilities,
    payoffs: payoffs,
    revealed: revealed,
    reveal_order: reveal_order,
    mu: mus,
    sigma: sigmas,
    PRs: PRs
    }
    
    return decision_problem
}


//generate the contents of the MouseLab table from the probabilities and payoffs
function generateMatrices(probabilities,payoffs,mu,sigma){
    
    clickCost = 0;
    nr_observations = 0;
    
    nr_outcomes=probabilities.length;
    nr_gambles=payoffs[0].length;
    
    nr_rows=nr_outcomes;
    nr_columns=nr_gambles+1;
    
    //initialize arrays
    name_matrix=new Array();
    inside_matrix=new Array();
    outside_matrix=new Array();
    active_matrix=new Array();

    name_matrix[0]=new Array();
    inside_matrix[0]=new Array();
    outside_matrix[0]=new Array();
    active_matrix[0]=new Array();
    
    RowOut=new Array();
    RowOut[0]=0;
    //outcome rows
    for (r=0; r<nr_rows;r++){ 
            //Probabilities
            name_matrix[r]=[num_abc(r+1)+"0"];
        if (r==0){
            outside_matrix[r]=[Math.round(100*probabilities[r])+"<font color='yellow'> YELLOW</font>"];// x "+num_abc(r)];
        }
        else if (r==1){
            outside_matrix[r]=[Math.round(100*probabilities[r])+" <font color='brown'> BROWN</font>"];//<svg height='100' width='100'><circle cx='50' cy='50' r='10' fill='red'/></svg>"];// x "+num_abc(r)];
        }
        else if (r==2){
            outside_matrix[r]=[Math.round(100*probabilities[r])+" <font color='blue'> BLUE</font>"];// x "+num_abc(r)];
        }
        else if (r==3){
            outside_matrix[r]=[Math.round(100*probabilities[r])+" <font color='purple'> PURPLE</font>"];// x "+num_abc(r)];
        }
            
            inside_matrix[r]=[Math.round(100*probabilities[r])/100+"%"];
            active_matrix[r]=[false];
            RowOut[r]=r;
        for (c=1;c<nr_columns;c++){

            name_matrix[r][c]=num_abc(r)+c;
            outside_matrix[r][c]="?"; //[Math.round(100*PR)/100];
            inside_matrix[r][c]=payoffs[r][c-1];
            active_matrix[r][c]=true;   
        }
    }
      
    statecont=active_matrix;
    tagcont=name_matrix;
    txtcont=inside_matrix;
    boxcont=outside_matrix;
        
    matrices={
        outside: outside_matrix,
        inside: inside_matrix,
        active: active_matrix,
        names: name_matrix,
        newly_revealed: []
    }
    
    return matrices;

}

function update_pseudorewards(matrices,dp){
    
    if (hasGambled==true){
        return matrices;
    }
    if (matrices.newly_revealed.length==2){
        r = matrices.newly_revealed[0];
        c = matrices.newly_revealed[1];
        if (dp.revealed[r][c]==false){
            sample_nr++;
            end_RT();
            start_RT();
            RTs[trial_nr_total-1][sample_nr-1] = RT;
        }
        else{
            return matrices;
        }
        dp.revealed[r][c] = true;
        dp.reveal_order[r][c] = sample_nr;
        clickCost++
        nr_observations++
        if (clickCost<10){
            $("#ClickCost").html(["Click cost: $0.0"+clickCost]);
        }
        else{
            $("#ClickCost").html(["Click cost: $0."+clickCost]);
        }
        $("#ClickCost").fadeIn();
    }
    
    decision_problem = dp; // make sure the update is global
    
    $("#MouseLabTable").html('')
    $("#MouseLabTable").html(mouselab_table(matrices.names,matrices.outside,matrices.inside,matrices.active));
    hInd = 0;
    for (r=0; r<dp.revealed.length;r++){ 
        for (c=0;c<dp.revealed[0].length;c++){
            if (dp.revealed[r][c]){
                hInd++
                fn = matrices.names[r][c+1]
                eval("HandleTxt"+hInd+"=document.all['"+fn+"_box"+"']");
                eval("HandleBox"+hInd+"=document.all['"+fn+"_box"+"']");
                eval("delay=window.setTimeout(\"HandleTxt"+hInd+".style.visibility='visible';HandleBox"+hInd+".style.visibility='hidden';\",dtime)");
            }
        }
    }
    
    return matrices;
}


function generate_feedback(probabilities,payoffs,chosen_gamble){
//This function simulated the outcome of the chosen gamble
        
    //Simulate outcome
    sampled_outcome=sampleDiscreteDistribution(probabilities);        
    
    if (chosen_gamble>0){
        //Determine payoff of chosen gamble for simulated outcome    
        payoff=payoffs[sampled_outcome][chosen_gamble-1];
    }
    else{
        //The participant chose "No thanks!"
        payoff=0;
    }
    
    return payoff;    
}

function hasChosen(gamble){
    end_RT();
    end_RTtrial();
    trialTime[trial_nr_total-1] = RTtrial;
    RTs[trial_nr_total-1][sample_nr] = RT;
    feedback[block_nr-1][trial_nr-1]=generate_feedback(decision_problem.probabilities,decision_problem.payoffs,gamble);
    nr_points+=feedback[block_nr-1][trial_nr-1];
    outcomes[trial_nr_total-1] = parseFloat(feedback[block_nr-1][trial_nr-1])
    observations[trial_nr_total-1] = nr_observations

    if (sampled_outcome==0){
        ball_html="<font color='yellow'>YELLOW.</font>"
    }
    else if (sampled_outcome==1){
        ball_html="<font color='brown'>BROWN.</font>"
    }
    else if (sampled_outcome==2){
        ball_html="<font color='blue'>BLUE.</font>"
    }
    else if (sampled_outcome==3){
        ball_html="<font color='purple'>PURPLE.</font>"
    }
    if (with_feedback[block_nr-1]){
        $("#NrPoints").html(nr_points)
        //alert("Gamble "+gamble+" has been chosen. Payoff: "+payoff)
        
        if (feedback[block_nr-1][trial_nr-1]>0){
            net_pay = feedback[block_nr-1][trial_nr-1] - clickCost/100
            net_pay = parseFloat(net_pay).toFixed(2);
            netPay[trial_nr_total-1] = net_pay
            if (isFullyRevealed==1){
                outcome_html=["The sampled ball is: "+ball_html+"&nbsp;&nbsp;&nbsp;You won <font color='green'><b>$"+feedback[block_nr-1][trial_nr-1]+"</b></font>"]
                clickCost_html = [""]
            }
            else{
                outcome_html=["The sampled ball is: "+ball_html+"&nbsp;&nbsp;&nbsp;You won $"+feedback[block_nr-1][trial_nr-1]]
                clickCost_html = ["Net earnings (winning minus click costs): <font color='green'><b>$"+net_pay+"</b></font>"]
            }
            //win_sound.play();
        }
        else{ if (feedback[block_nr-1][trial_nr-1]<0){
            outcome_html="<font color='red'><b>$"+feedback[block_nr-1][trial_nr-1]+"</b></font>"
            //loss_sound.play();
            clickCost_html = ""
        }
        else{
            outcome_html="0";
            clickCost_html = ""
        }
        }
        
        $("#Outcome").html(outcome_html);
        $("#Outcome").fadeIn();
        $("#ClickCost").html(clickCost_html);
        $("#ClickCost").fadeIn();
//        setTimeout(function(){$("#Outcome").fadeOut();},1800);         
        
    }
    else{
        $("#NrPoints").html("??")
        $("#Outcome").html("")
        $("#ClickCost").html("")
        $("#Timer").html("")
    }
}

function saveAnswers(){
                
    //bonus = outcomes[Math.floor(Math.random()*trial_nr_total)+1-1];
    end_experimentTime();
    var foo1 = [];
    for (var i = 0; i < nr_trials/2; i++) {
        foo1.push(i);
    }
    var foo2 = [];
    for (var i = nr_trials/2; i < nr_trials; i++) {
        foo2.push(i);
    }
    tmp1 = shuffle(foo1);
    bonus1 = netPay[tmp1[0]];
    tmp2 = shuffle(foo2);
    bonus2 = netPay[tmp2[0]];
    bonus = Math.max(0,(Math.max(bonus1)+Math.max(bonus2))/2);
    bonus = parseFloat(bonus).toFixed(2);
    
//    bonus = Math.max(0,netPay[Math.floor(Math.random()*trial_nr_total)+1-1]);
    
    basic_info={
        nr_trials:nr_trials,
        nr_blocks:nr_blocks,
        block1_stakes: block1_stakes,
        block2_stakes: block2_stakes,
        payoff_range1: payoff_range1,
        payoff_range2: payoff_range2,
        payoff_mu1: payoff_mu1,
        payoff_std1: payoff_std1,
        payoff_mu2: payoff_mu2,
        payoff_std2: payoff_std2,
        isHighCompensatory: isHighCompensatory,
        failed_quiz: failed_quiz,
        isFullyRevealed: isFullyRevealed
    }
    
    data={           
        bonus: bonus,            
        decisions: choices,
        feedback: feedback,/////
        decision_problems: decision_problems,
        //process_data: process_data,
        outcomes: outcomes,
        netPay: netPay,
        observations: observations,
        trialTime: trialTime,
        RTs: RTs,
        instructionQuizTime: quizTime,
        experimentTime: experimentTime,
        basic_info: basic_info/////
    }
                
    $("#bonus").html(bonus);
    $("#finishedFeedback").hide();
    $("#Test").hide();
    $("#Debriefing").show();        
    setTimeout(function() {turk.submit(data)},7500);
}



////////////////////////////////////
//Functions of general utility    //
////////////////////////////////////

function stickBreaking(nr_outcomes,alpha){
    //generate outcome probabilities by a stick-breaking process    
    var probabilities=new Array();
    
    var cumsum=0;
        for (p=0; p<nr_outcomes-1; p++){
            if (alpha==1){ //sample stick proportions from uniform distribution
                probabilities[p]=Math.random()*(1-cumsum);
            }
            else{
                probabilities[p]=jStat.beta.sample(1,alpha)*(1-cumsum);
            }
            cumsum+=probabilities[p];
        }
        probabilities[nr_outcomes-1]=1-cumsum;

    return probabilities;
}
    
function num_abc(number){
    return String.fromCharCode(number+65);
}

function sampleDiscreteDistribution(probabilities){
    //sample outcome from outcome distribution by the inverse distribution function method
    //returns an integer between 0 and probabilities.length-1
    var p=Math.random();
    var nr_outcomes=probabilities.length;
    var cdf=new Array(nr_outcomes);
    cdf[0]=probabilities[0];
    if (cdf[0]>=p){
        first_greater=0;
    }
    else{
        first_greater=nr_outcomes;
    }
    for (o=1;o<nr_outcomes;o++){
        cdf[o]=cdf[o-1]+probabilities[o];
        
        if (cdf[o]>=p && o<first_greater){
            first_greater=o;
        }
    }
    
    var sampled_outcome=first_greater;
    return sampled_outcome;
}

function computeExpectedValues(probabilities,payoffs){
    var nr_outcomes=probabilities.length;
    var nr_gambles=payoffs[0].length;
    
    var EVs=new Array(nr_gambles);
        var max_EV=Number.NEGATIVE_INFINITY;
        var best_gamble=0;
        for (g=0;g<nr_gambles;g++){
            EVs[g]=0;
            for (o=0;o<nr_outcomes;o++){
                EVs[g]+=probabilities[o]*payoffs[o][g];            
            }
            if (EVs[g]>max_EV){
                best_gamble=g;
                max_EV=EVs[g];
            }
        }
    
    EUT={
        values: EVs,
        choice: best_gamble
    }
    
    return EUT;
}

function makeCompensatory(probabilities,payoffs){
    
    var nr_outcomes=probabilities.length;
    var nr_gambles=payoffs[0].length;
    
    var EUT=computeExpectedValues(probabilities,payoffs);
    
    var MPO=0; //Most Probable Outcome
    var max_probability=0;
        
    for (o=0;o<nr_outcomes;o++){
        if (probabilities[o]>max_probability){
            max_probability=probabilities[o];
            MPO=o;
        }
    }
    
    var TTB_decision=0;
    for (g=0;g<nr_gambles;g++){
        if(payoffs[MPO][g]>payoffs[MPO][TTB_decision]){
            TTB_decision=g;
        }
    }
 
    if (EUT.choice==TTB_decision && nr_gambles>1 && nr_outcomes>1){   
    //Make the best_gamble worse than the second best one.
        //0. Determine the second best gamble
        better_gamble=0; second_highest_EV=-high_range;
        for (g=0;g<nr_gambles;g++){
            if (g!=TTB_decision && EUT.values[g]>second_highest_EV){
                second_highest_EV=EUT.values[g];
                better_gamble=g;
            }
        }
        
        //Reduce the gap between the highest and the second highest probability
        //a. Determine the second highest probability
        second_largest_prob=0;
        for (o=0;o<nr_outcomes;o++){
                if (o!=MPO && probabilities[o]>second_largest_prob){
                    second_largest_prob=probabilities[o];
                }
        }   
        //b. Reduce the highest probability towards the second highest probability
        probabilities[MPO]=second_largest_prob+0.1;
        delta_p=max_probability-probabilities[MPO];
        //c. Distribute the decrement equally across all outcomes
        for (o=0;o<nr_outcomes;o++){
            probabilities[o]+=delta_p/nr_outcomes;
        }
        //d. Round probabilities
        probabilities=roundProbabilities(probabilities);
        
        payoffs[MPO][better_gamble]=payoffs[MPO][TTB_decision]-1 -0.05*Math.random()*(payoffs[MPO][TTB_decision]-payoffs[MPO][better_gamble]);
        EUT=computeExpectedValues(probabilities,payoffs);
        
        //If the gamble with the best outcome for the most probable outcome is still superior, then reduce its payoffs for the other outcomes.
        delta_EV=EUT.values[TTB_decision]-EUT.values[better_gamble];
        if (delta_EV>=0){
            boost=delta_EV/(1-max_probability)+1;
            for (o=0;o<nr_outcomes;o++){
                if (o!=MPO){
                    payoffs[o][better_gamble]+=boost;
                    payoffs[o][TTB_decision]-=boost;
                }
            }
        }
        
        EUT=computeExpectedValues(probabilities,payoffs);
        
        if (EUT.choice==TTB_decision){
            alert('Correction failed.')
        }
}
        gamble = {
            payoffs: payoffs,
            probabilities: probabilities
        }

        return gamble;
}

function TTBChoice(gamble){
        var unique_max=false;
        var highest_probability=0;
        var MPO=-1;
        for (o=0;o<nr_outcomes;o++){
            if (gamble.probabilities[o]>highest_probability){
                MPO=o;
                highest_probability=gamble.probabilities[o];
                unique_max=true;
            }
            else{
                if (gamble.probabilities[o]==highest_probability){
                    unique_max=false;
                }
            }
        }
        
        TTB_decision=-1;
        best_outcome=Number.NEGATIVE_INFINITY;
        for (g=0;g<nr_gambles;g++){
            if (gamble.payoffs[MPO][g]>best_outcome){
                best_outcome=gamble.payoffs[MPO][g];
                TTB_decision=g;
            }
        }
    return TTB_decision;
}

function roundProbabilities(probabilities){
    //Ensure that all probabilities are >=0.01 and round probabilities to full %
    sum_of_probabilities=0;
    max_probability=0;
    i_small=probabilities.length-1;
    i_large=0;
    for (p=0;p<probabilities.length;p++){
        
        probabilities[p]=Math.round(100*probabilities[p])/100;
                
        if (probabilities[p]<0.01){
            probabilities[p]=0.01;
            i_small=p;
        }
        
        sum_of_probabilities+=probabilities[p];
        
        if (probabilities[p]>max_probability){
            i_large=p;
            max_probability=probabilities[p];
        }
    }
    
    if (sum_of_probabilities>1){
        probabilities[i_large]-=Math.round(100*(sum_of_probabilities-1))/100;
    }
    else{
        probabilities[i_small]+=Math.round(100-100*sum_of_probabilities)/100;
    }
    
    return probabilities;
}

function min2d(numbers){
    
    minimum=_.min(numbers[0]);
    for (r=1;r<numbers.length;r++){
        if (_.min(numbers[r])<minimum){
            minimum=_.min(numbers[r]);
        }
    }
    
    return minimum;
}

function max2d(numbers){
    
    maximum=_.max(numbers[0]);
    for (r=1;r<numbers.length;r++){
        if (_.max(numbers[r])>maximum){
            maximum=_.max(numbers[r]);
        }
    }    
    return maximum;
}

String.prototype.repeat = function( num )
{
    return new Array( num + 1 ).join( this );
}

function cloneArray2D(array2d){
    copy=new Array(array2d.length);
    
    for (i=0;i<array2d.length;i++){
        copy[i]=new Array(array2d[i].length);
        for (j=0;j<array2d[i].length;j++){
            copy[i][j]=array2d[i][j];
        }
    }

    return copy;
}

function randn_trunc(mu,sd,range){
    ii = 1;
    while (ii == 1 || randnn < range[0] || randnn > range[1]){
        ii = 0;
        u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
        v = 1 - Math.random();
        randnn = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        randnn = randnn*sd+mu;
    }
    return randnn
}

function randn(mu,sd){
    u = 1 - Math.random(); // Subtraction to flip [0, 1) to (0, 1].
    v = 1 - Math.random();
    randnn = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    randnn = randnn*sd+mu;
    return randnn
}

function make_dp_copy(dp){
    var dp_copy={
        probabilities: dp.probabilities.slice(),
        payoffs: dp.payoffs.slice(),
        revealed: dp.revealed.slice(),
        mu: dp.mu.slice(),
        sigma: dp.sigma.slice(),
        PRs: dp.PRs.slice()
    }
    
    return dp_copy
}

function shuffle(a) {
    var j, x, i;
    for (i = a.length; i; i--) {
        j = Math.floor(Math.random() * i);
        x = a[i - 1];
        a[i - 1] = a[j];
        a[j] = x;
    }
    return a;
}

function start_RT() {
  startTime_RT = new Date();
};

function end_RT() {
  endTime_RT = new Date();
  RT = (endTime_RT - startTime_RT)/1000;
//  var RT = endTime_RT - startTime_RT; //in ms
//  // strip the ms
//  RT /= 1000;

//  // get seconds 
//  var seconds = Math.round(timeDiff);
//  console.log(seconds + " seconds");
}

function start_RTtrial() {
  startTime_trial = new Date();
};

function end_RTtrial() {
  endTime_trial = new Date();
  RTtrial = (endTime_trial - startTime_trial)/1000; //in ms
//  var trial_time = endTime_trial - startTime_trial; //in ms
//  // strip the ms
//  trial_time /= 1000;

//  // get seconds 
//  var seconds = Math.round(timeDiff);
//  console.log(seconds + " seconds");
}

function fillArray(value, len) {
  if (len == 0) return [];
  var a = [value];
  while (a.length * 2 <= len) a = a.concat(a);
  if (a.length < len) a = a.concat(a.slice(0, len - a.length));
  return a;
}

function start_quizTime() {
  startTime_quiz = new Date();
};

function end_quizTime() {
  endTime_quiz = new Date();
  quizTime = (endTime_quiz - startTime_quiz)/1000;
}

function start_experimentTime() {
  startTime_experiment = new Date();
};

function end_experimentTime() {
  endTime_experiment = new Date();
  experimentTime = (endTime_experiment - startTime_experiment)/1000;
}

/////////////////////////
// end of code //
/////////////////////////