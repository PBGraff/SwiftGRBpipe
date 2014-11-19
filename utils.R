gridToXY <- function(grid) {
        x <- (grid %% 7 - 3) / 2
        y <- (as.integer(grid/7) - 2) / 3
        r <- sqrt(x^2 + y^2)
        phi <- atan2(y,x)
        c(r,phi)
}

