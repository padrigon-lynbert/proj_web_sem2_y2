<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/Storage/Boostrap/css/bootstrap.min.css">
    <link rel="stylesheet" href="/Storage/Dashboard/CSS/style.css">
    <title>AdminPage</title>
</head>
<body>

    <?php
        include "connection.php";
    ?>

    <!-- NAVBAR -->
    <nav class="navbar navbar-expand-lg bg-white sticky-top">
        <div class="container">
            <a class="navbar-brand" href="#">
                <img src="/Storage/Dashboard/Images/s-w.png" alt="">
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#hero">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#about">Info</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#services">Training</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#portfolio">Portfolio</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#reviews">Reviews</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#team">Leaders</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#contact">Contact</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#blog">Blog</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="Admin-Dashboard.php">Dashboard</a>
                    </li>
                </ul>
                <a href="/Storage/Login/login.php" class="btn btn-brand ms-lg-3">Download</a>
            </div>
        </div>
    </nav>




    <div class="container">
        <button class="btn btn-primary my-5"><a href="user.php" class="text-light">Add Employee</a></button>
    </div>



    <table class="table my-5" style="margin-left: 50px;">
        <thead>
            <tr>
            <th scope="col">Index #</th>
            <th scope="col">Fname</th>
            <th scope="col">Lname</th>
            <th scope="col">current_position</th>
            <th scope="col">Promoted to</th>
            </tr>
        </thead>

        <tbody>
            <?php
                $sql = "select * from `employee`";
                $result = mysqli_query($connection, $sql);

                if($result)
                {
                    while($row = mysqli_fetch_assoc($result))
                    {
                        $id = $row['id'];
                        $fname = $row['fname'];
                        $lname = $row['lname'];
                        $current_position = $row['current_position'];
                        $promotion_to = $row['promotion_to'];
                        echo '<tr>
                            <th scope="row">'.$id.'</th>
                            <td>'.$fname.'</td>
                            <td>'.$lname.'</td>
                            <td>'.$current_position.'</td>
                            <td>'.$promotion_to.'</td>

                            <td>
                                <button class="btn btn-primary">
                                    <a class="text-dark" href="update.php?updateid= '.$id.' ">Update</a>
                                </button>
                                <button class="btn btn-danger">
                                    <a href="delete.php?deleteid= '.$id.'" class="text-dark">Delete</a>
                                </button>

                            </td>
                        </tr>';
                    }
                   
                    // echo $row['lname'];
                }
            ?>

            
           
        </tbody>
    </table>


    <!-- REVIEW -->
<section id="reviews" class="section-padding bg-light">
    <div class="container">
        <div class="row">
            <div class="col-12 text-center" data-aos="fade-down" data-aos-delay="150">
                <div class="section-title">
                    <h1 class="display-4 fw-semibold">Testimonials</h1>
                    <div class="line"></div>
                    <p>We love to craft digital experiances for brands rather than crap and more lorem ipsums and do crazy skills</p>
                </div>
            </div>
        </div>
        <div class="row gy-5 gx-4">
            <div class="col-lg-4 col-sm-6" data-aos="fade-down" data-aos-delay="150">
                <div class="review">
                    <div class="review-head p-4 bg-white theme-shadow">
                        <div class="text-warning">
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                        </div>
                        <p>Amazing theme ipsum dolor sit amet consectetur adipisicing elit. Assumenda eum animi rerum ipsam impedit dicta voluptatem.</p>
                    </div>
                    <div class="review-person mt-4 d-flex align-items-center">
                        <img class="rounded-circle" src="/Storage/Login/images/user.png" style="height: 50px;">
                        <div class="ms-3">
                            <h5>Dianne Russell</h5>
                            <small>UX Architect</small>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-lg-4 col-sm-6" data-aos="fade-down" data-aos-delay="150">
                <div class="review">
                    <div class="review-head p-4 bg-white theme-shadow">
                        <div class="text-warning">
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                        </div>
                        <p>Amazing theme ipsum dolor sit amet consectetur adipisicing elit. Assumenda eum animi rerum ipsam impedit dicta voluptatem.</p>
                    </div>
                    <div class="review-person mt-4 d-flex align-items-center">
                        <img class="rounded-circle" src="/Storage/Login/images/user.png" style="height: 50px;">
                        <div class="ms-3">
                            <h5>Dianne Russell</h5>
                            <small>UX Architect</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4 col-sm-6" data-aos="fade-down" data-aos-delay="150">
                <div class="review">
                    <div class="review-head p-4 bg-white theme-shadow">
                        <div class="text-warning">
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-half.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                        </div>
                        <p>Amazing theme ipsum dolor sit amet consectetur adipisicing elit. Assumenda eum animi rerum ipsam impedit dicta voluptatem.</p>
                    </div>
                    <div class="review-person mt-4 d-flex align-items-center">
                        <img class="rounded-circle" src="/Storage/Login/images/user.png" style="height: 50px;">
                        <div class="ms-3">
                            <h5>Dianne Russell</h5>
                            <small>UX Architect</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4 col-sm-6" data-aos="fade-down" data-aos-delay="150">
                <div class="review">
                    <div class="review-head p-4 bg-white theme-shadow">
                        <div class="text-warning">
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                        </div>
                        <p>Amazing theme ipsum dolor sit amet consectetur adipisicing elit. Assumenda eum animi rerum ipsam impedit dicta voluptatem.</p>
                    </div>
                    <div class="review-person mt-4 d-flex align-items-center">
                        <img class="rounded-circle" src="/Storage/Login/images/user.png" style="height: 50px;">
                        <div class="ms-3">
                            <h5>Dianne Russell</h5>
                            <small>UX Architect</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4 col-sm-6" data-aos="fade-down" data-aos-delay="150">
                <div class="review">
                    <div class="review-head p-4 bg-white theme-shadow">
                        <div class="text-warning">
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                        </div>
                        <p>Amazing theme ipsum dolor sit amet consectetur adipisicing elit. Assumenda eum animi rerum ipsam impedit dicta voluptatem.</p>
                    </div>
                    <div class="review-person mt-4 d-flex align-items-center">
                        <img class="rounded-circle" src="/Storage/Login/images/user.png" style="height: 50px;">
                        <div class="ms-3">
                            <h5>Dianne Russell</h5>
                            <small>UX Architect</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4 col-sm-6" data-aos="fade-down" data-aos-delay="150">
                <div class="review">
                    <div class="review-head p-4 bg-white theme-shadow">
                        <div class="text-warning">
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-fill.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                            <i class="ri-star-fill"><img src="/Storage/Dashboard/icons/star-empty.svg" alt=""></i>
                        </div>
                        <p>Amazing theme ipsum dolor sit amet consectetur adipisicing elit. Assumenda eum animi rerum ipsam impedit dicta voluptatem.</p>
                    </div>
                    <div class="review-person mt-4 d-flex align-items-center">
                        <img class="rounded-circle" src="/Storage/Login/images/user.png" style="height: 50px;">
                        <div class="ms-3">
                            <h5>Dianne Russell</h5>
                            <small>UX Architect</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>



    
 <!-- TEAM -->
 <section id="team" class="section-padding">
    <div class="container">
        <div class="row">
            <div class="col-12 text-center" data-aos="fade-down" data-aos-delay="150">
                <div class="section-title">
                    <h1 class="display-4 fw-semibold">Core Leaders</h1>
                    <div class="line"></div>
                    <p>Milk pls.</p>
                </div>
            </div>
        </div>
        <div class="row g-4 text-center ">
            <div class="col-md-3" data-aos="fade-down" data-aos-delay="150">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img src="/Storage/Dashboard/Images/cats/c2.jpg" style="height: 250px;" alt="">
                    </div>
                    <div class="team-member-content">
                        <!-- <h4 class="text-white">Ninja Engineer, Chibi</h4> -->
                        <h4 class="text-white">Ninja Chibi</h4>
                        <!-- <p class="mb-0 text-white">Data Science and Machine Learning</p> -->
                        <p class="mb-0 text-white">CEO</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3" data-aos="fade-down" data-aos-delay="250">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img src="/Storage/Dashboard/Images/cats/c1.jpg" style="height: 250px;" alt="">

                    </div>
                    <div class="team-member-content">
                        <!-- <h4 class="text-white">Rockstar Developer, Kibi</h4> -->
                        <h4 class="text-white">Rockstar Kibi</h4>
                        <p class="mb-0 text-white">Chief Director</p>
                        <!-- <p class="mb-0 text-white">Webflow Artist</p> -->
                    </div>
                </div>
            </div>
            <div class="col-md-3" data-aos="fade-down" data-aos-delay="350">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img src="/Storage/Dashboard/Images/cats/c3.jpg" style="height: 250px;" alt="">

                    </div>
                    <div class="team-member-content">
                        <h4 class="text-white">Milk Muddy</h4>
                        <!-- <h4 class="text-white">Milk Barrista, Muddy</h4> -->
                        <!-- <p class="mb-0 text-white">Coffee Maker</p> -->
                        <p class="mb-0 text-white">Chief Operating Manager</p>
                    </div>
                </div>
            </div>

            <div class="col-md-3" data-aos="fade-down" data-aos-delay="450">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img src="/Storage/Dashboard/Images/cats/c4.jpg" style="height: 250px;" alt="">

                    </div>
                    <div class="team-member-content">
                        <h4 class="text-white">Imu</h4>
                        <!-- <h4 class="text-white">Team Leader, Imu</h4> -->
                        <!-- <p class="mb-0 text-white">Leader</p> -->
                        <p class="mb-0 text-white">Chief Operating Officer</p>
                    </div>
                </div>
            </div>

        </div>
    </div>
</section>




<!-- BLOG -->
<section id="blog" class="section-padding">
    <div class="container">
        <div class="row">
            <div class="col-12 text-center" data-aos="fade-down" data-aos-delay="150">
                <div class="section-title">
                    <h1 class="display-4 fw-semibold">Recent News & Articles</h1>
                    <div class="line"></div>
                    <p>We love to craft digital experiances for brands rather than crap and more lorem ipsums and do crazy skills</p>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-md-4" data-aos="fade-down" data-aos-delay="150">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img style="height: 500px;" src="/Storage/Dashboard/Images/news/Morgan Stanley’s CEO Gives Himself an A-Minus.jpg" alt="">                        </div>
                    <h5 class="mt-4">Web Design 2022</h5>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Sit sequi quos magni!</p>
                    <a href="#">Read More</a>
                </div>
            </div>
            <div class="col-md-4" data-aos="fade-down" data-aos-delay="250">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img style="height: 500px;" src="/Storage/Dashboard/Images/news/CEO Today Magazine.jpg" alt="">
                    </div>
                    <h5 class="mt-4">Web Design 2022</h5>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Sit sequi quos magni!</p>
                    <a href="#">Read More</a>
                </div>
            </div>
            <div class="col-md-4" data-aos="fade-down" data-aos-delay="350">
                <div class="team-member image-zoom">
                    <div class="image-zoom-wrapper">
                        <img style="height: 500px;" src="/Storage/Dashboard/Images/news/97 Best Business Motivational Quotes & Memes _ CLMB Marketing.jpg" alt="">
                    </div>
                    <h5 class="mt-4">Web Design 2022</h5>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Sit sequi quos magni!</p>
                    <a href="#">Read More</a>
                </div>
            </div>
        </div>
    </div>
</section>



   


    


    

      
<!-- FOOTER -->
<footer class="bg-dark">
    <div class="footer-top">
        <div class="container">
            <div class="row gy-5">
                <div class="col-lg-3 col-sm-6">
                    <a href="#"><img src="/Storage/Dashboard/Images/s-w.png" alt=""></a>
                    <div class="line"></div>
                    <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Exercitationem, hic!</p>
                    <div class="social-icons">
                        <a href="#"><i class="ri-twitter-fill"><img class="icon-img3" src="/Storage/Dashboard/Images/footer/github.svg" alt=""></i></a>
                        <a href="#"><i class="ri-twitter-fill"><img class="icon-img3" src="/Storage/Dashboard/Images/footer/facebook.svg" alt=""></i></a>
                        <a href="#"><i class="ri-twitter-fill"><img class="icon-img3" src="/Storage/Dashboard/Images/footer/instagram.svg" alt=""></i></a>
                        <a href="#"><i class="ri-twitter-fill"><img class="icon-img3" src="/Storage/Dashboard/Images/footer/info-circle.svg" alt=""></i></a>
                    </div>
                </div>
                <div class="col-lg-3 col-sm-6">
                    <h5 class="mb-0 text-white">SERVICES</h5>
                    <div class="line"></div>
                    <ul>
                        <li><a href="#">Machine Learning</a></li>
                        <li><a href="#">UI UX Destroyer</a></li>
                        <li><a href="#">Briefs Model</a></li>
                        <li><a href="#">Clown</a></li>
                    </ul>
                </div>
                <div class="col-lg-3 col-sm-6">
                    <h5 class="mb-0 text-white">ABOUT</h5>
                    <div class="line"></div>
                    <ul>
                        <li><a href="#blog">Blog</a></li>
                        <li><a href="#services">Services</a></li>
                        <li><a href="#">Company</a></li>
                        <li><a href="#">Career</a></li>
                    </ul>
                </div>
                <div class="col-lg-3 col-sm-6">
                    <h5 class="mb-0 text-white">CONTACT</h5>
                    <div class="line"></div>
                    <ul>
                        <li>New York, NY 3300</li>
                        <li>(414) 586 - 3017</li>
                        <li>padirigonlynbert@services.com</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    <div class="footer-bottom">
        <div class="container">
            <div class="row g-4 justify-content-between">
                <div class="col-auto">
                    <p class="mb-0">© Copyright Succession | All Rights Reserved.</p>
                </div>
                <div class="col-auto">
                    <p class="mb-0"> By <a href="https://">Padrigon, L.S. Orilla</a></p>
                </div>
            </div>
        </div>
    </div>
</footer>








    <script src="/Storage/Boostrap/js/bootstrap.bundle.min.js"></script>
</body>
</html>