package LibraVDB::Collection;
use strict;
use warnings;
use LibraVDB qw( :all );

sub new {
    my ($class, $db_id, $col_id, $name, $dimension) = @_;
    my $self = {
        db_id     => $db_id,
        col_id    => $col_id,
        name      => $name,
        dimension => $dimension,
    };
    bless $self, $class;
    return $self;
}

sub insert {
    my ($self, $id, $vector, $metadata) = @_;
    $metadata //= '{}';
    
    if (scalar(@$vector) != $self->{dimension}) {
        die "Vector dimension mismatch\n";
    }
    
    my $err = LibraVDB::InsertVector($self->{col_id}, $id, $vector, $self->{dimension}, $metadata);
    _check_error($err);
}

sub insert_batch {
    my ($self, $ids, $vectors, $metadata) = @_;
    my $count = scalar(@$ids);
    
    if (scalar(@$vectors) != $count) {
        die "Length of ids and vectors must match\n";
    }
    if (defined $metadata && scalar(@$metadata) != $count) {
        die "Length of ids and metadata must match\n";
    }
    
    my @flat_vectors;
    my @metas;
    for my $i (0 .. $count - 1) {
        if (scalar(@{$vectors->[$i]}) != $self->{dimension}) {
            die "Vector dimension mismatch at index $i\n";
        }
        push @flat_vectors, @{$vectors->[$i]};
        push @metas, (defined $metadata ? $metadata->[$i] : '{}');
    }
    
    my $err = LibraVDB::InsertBatch(
        $self->{col_id},
        $ids,
        \@flat_vectors,
        $count,
        $self->{dimension},
        \@metas
    );
    _check_error($err);
}

sub search {
    my ($self, $vector, $k, $filter) = @_;
    $filter //= '{}';
    
    if (scalar(@$vector) != $self->{dimension}) {
        die "Vector dimension mismatch\n";
    }
    
    my $res = LibraVDB::QueryVector($self->{col_id}, $vector, $self->{dimension}, $k, $filter);
    return _extract_string($res);
}

sub scan {
    my ($self, $offset, $limit) = @_;
    $offset //= 0;
    $limit  //= 100;
    
    my $res = LibraVDB::ScanCollection($self->{col_id}, $offset, $limit);
    return _extract_string($res);
}

sub delete {
    my ($self, $id) = @_;
    my $err = LibraVDB::DeleteVector($self->{col_id}, $id);
    _check_error($err);
}

sub _check_error {
    my ($err_str) = @_;
    return unless defined $err_str;
    return if $err_str eq 'OK';
    
    if ($err_str =~ /^ERROR:/ || $err_str =~ /^error/i) {
        die "$err_str\n";
    }
}

sub _extract_string {
    my ($str) = @_;
    return "{}" unless defined $str;
    
    if ($str =~ /^ERROR:/ || $str =~ /^error/i) {
        die "$str\n";
    }
    return $str;
}

1;
