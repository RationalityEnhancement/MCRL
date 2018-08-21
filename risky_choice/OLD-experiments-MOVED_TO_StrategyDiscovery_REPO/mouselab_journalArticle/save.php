<?php

$trial_nr=1;
$trial_nr = $trial_nr+1;

print_r($_POST);

$url = 'http://localhost/MouseLab/test2.php';
//$data = array_push($data, 'last_trial', $_POST)
$data=$_POST;
    
$options = array(
    'http' => array(
        'header'  => "Content-type: application/x-www-form-urlencoded\r\n",
        'method'  => 'POST',
        'content' => http_build_query($data),
    ),
);
$context  = stream_context_create($options);
$html = file_get_contents($url, false, $context);

var_dump($html);

?>